import streamlit as st
import tools
import json
import os

# ==========================================
# 1. 初始化全局状态
# ==========================================

# 1.1 初始化模型字典
if "model_dict" not in st.session_state:
    st.session_state.model_dict = {
        "Geometry": {
            "Source": "Rhino_Import",
            "Zones": ["Office_South", "Office_North", "Meeting_Room"] 
        }
    }

# 1.2 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是 Energy Agent。\n让我们按顺序配置模型。\n\n第一步：请输入项目所在的**城市**（如：上海）。"}
    ]

# 1.3 初始化当前步骤
# 顺序：0=Location -> 1=Schedule -> 2=HVAC -> 3=Material -> 4=Done
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# ==========================================
# 2. 页面布局
# ==========================================
st.set_page_config(layout="wide", page_title="Energy Agent Demo")

col_chat, col_status = st.columns([2, 1])

# --- 右侧：状态监视器 & 自动保存反馈 ---
with col_status:
    st.subheader("🔍 模型实时状态")
    
    # 显示进度
    st.progress(st.session_state.current_step / 4)
    st.caption(f"Current Step: {st.session_state.current_step}")
    
    # 显示字典内容
    st.json(st.session_state.model_dict)

    # 【修改点】自动保存逻辑
    if st.session_state.current_step == 4:
        # 定义保存路径
        save_path = "building_model.json"
        
        # 写入文件
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.model_dict, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ 模型已自动保存！")
            st.info(f"文件路径: {os.path.abspath(save_path)}")
            st.caption("现在你可以运行 converter.py 来生成 IDF 了。")
            
        except Exception as e:
            st.error(f"保存失败: {e}")

# --- 左侧：聊天窗口 ---
with col_chat:
    st.subheader("💬 交互控制台")
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ==========================================
    # 3. 核心交互逻辑 (状态机模式)
    # ==========================================
    if user_input := st.chat_input("请输入..."):
        
        # 记录用户输入
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        response = ""
        step = st.session_state.current_step

        # --- Step 0: 地点 ---
        if step == 0:
            if "上海" in user_input or "北京" in user_input:
                city = "Shanghai" if "上海" in user_input else "Beijing"
                res = tools.set_location(city, st.session_state.model_dict)
                response = f"{res}\n\n**下一步**：这是什么类型的建筑？（请输入 '办公' 或 '住宅'）"
                st.session_state.current_step = 1 
            else:
                response = "请提供有效的城市名称（目前支持：上海、北京）。"

        # --- Step 1: 建筑类型 ---
        elif step == 1:
            if "办公" in user_input:
                res = tools.set_schedule("Office", st.session_state.model_dict)
                response = f"{res}\n\n**下一步**：请选择空调系统。（请输入 'VRF' 或 'VAV'）"
                st.session_state.current_step = 2
            elif "住宅" in user_input:
                res = tools.set_schedule("Residential", st.session_state.model_dict)
                response = f"{res}\n\n**下一步**：请选择空调系统。（请输入 'VRF' 或 'VAV'）"
                st.session_state.current_step = 2
            else:
                response = "请明确建筑类型（输入 '办公' 或 '住宅'）。"

        # --- Step 2: 空调 ---
        elif step == 2:
            if "VRF" in user_input.upper() or "VAV" in user_input.upper():
                sys_type = "VRF" if "VRF" in user_input.upper() else "VAV"
                res = tools.set_hvac(sys_type, st.session_state.model_dict)
                
                city = st.session_state.model_dict["Location"]["City"]
                opts = tools.get_material_options(city)
                
                response = f"{res}\n\n**下一步**：最后，请选择围护结构材料方案。\n{opts}"
                st.session_state.current_step = 3
            else:
                response = "请选择有效的系统类型（输入 'VRF' 或 'VAV'）。"

        # --- Step 3: 材料 ---
        elif step == 3:
            if "A" in user_input.upper() or "B" in user_input.upper():
                city = st.session_state.model_dict["Location"]["City"]
                choice = "Recommended" if "A" in user_input.upper() else "Economy"
                res = tools.set_material_choice(city, choice, st.session_state.model_dict)
                
                response = f"{res}\n\n🎉 **全部完成！** 配置文件已自动保存到项目目录。"
                st.session_state.current_step = 4
            else:
                response = "请通过输入 'A' 或 'B' 来选择材料方案。"

        # --- Step 4: 完成 ---
        elif step == 4:
            response = "模型已构建完成。如需修改，请刷新页面重新开始。"

        # ------------------------------
        
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()