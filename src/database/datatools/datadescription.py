import sqlite3

def _gen_description_material(data:list):
    des = f"这是energyplus中的standard_materials下的材料数据，在数据库中的id为{data[0]}，名称为{data[1]}，该材料数据来源被应用于纬度为{data[2]}经度为{data[3]}的地区，其用于建筑类型为{data[4]}，粗糙度为{data[5]} ，厚度为{data[6]}m，导热性为{data[7]}W/(m*k)，密度为{data[8]}kg/m3，比热容为{data[9]}J/(kg*k),热吸收率为{data[10]}，太阳辐射吸收率为{data[11]}J/(kg*k)，可见光吸收率为{data[12]}。"
    return des

def _gen_description_nomass_material(data:list):
    des = f"这是energyplus中的no_mass_materials下的材料数据，在数据库中的id为{data[0]}，名称为{data[1]}，该材料数据来源被应用于纬度为{data[2]}经度为{data[3]}的地区，其用于建筑类型为{data[4]}，粗糙度为{data[5]} ，热阻为{data[6]}m^2*K/W，热吸收率为{data[7]}，太阳辐射吸收率为{data[8]}J/(kg*k)，可见光吸收率为{data[9]}。"
    return des

def _gen_description_construction(data:list):
    des = f"这是energyplus中的constructions下的构造数据，在数据库中的id为{data[0]}，名称为{data[1]}，该构造数据来源被应用于纬度为{data[2]}经度为{data[3]}的地区，其用于建筑类型为{data[4]}，材料层依次为：{', '.join([str(layer) for layer in data[5:25] if layer is not None])}，layer中的数据指向all_materials表中对应的材料id。"
    return des

def _gen_description_schedule_type_limits(data:list):
    des = f"这是energyplus中的schedule_type_limits下的时间表类型限制数据，在数据库中的id为{data[0]}，名称为{data[1]}，该时间表类型限制数据来源被应用于纬度为{data[2]}经度为{data[3]}的地区，其用于建筑类型为{data[4]}，下限值为{data[5]}，上限值为{data[6]}，数值类型为{data[7]}，单位类型为{data[8]}。"
    return des

def _gen_description_schedule_compact(data:list):
    des = f"这是energyplus中的schedule_compact下的时间表压缩数据，在数据库中的id为{data[0]}，名称为{data[1]}，该时间表压缩数据来源被应用于纬度为{data[2]}经度为{data[3]}的地区，其用于建筑类型为{data[4]}，时间表类型限制名称为{data[5]}，时间表安排为：{' | '.join([str(layer) for layer in data[6:-2] if layer is not None])}，计划表中through表示到一年中前一个through到这一天前执行后面写的计划，for表示对于一周中哪些天生效后面写的计划，until表示一天中上一个until到这一个时刻保持下一个数值，如果是第一个through或者until则表示从一年的第一天到这一天或者一天的开始到这一时刻。"
    return des

def _gen_description_sizingperiod_designday(data:list):
    des = f"这是energyplus中的sizingperiod_designday下的设计日数据，在数据库中的id为{data[0]}，名称为{data[1]}，该设计日数据来源被应用于纬度为{data[2]}经度为{data[3]}的地区，其用于建筑类型为{data[4]}，月份为{data[5]}，日期为{data[6]}，日期类型为{data[7]}，最大干球温度为{data[8]}℃，日干球温度范围为{data[9]}℃，干球温度范围修正器类型为{data[10]}，干球温度范围修正日程表名称为{data[11]}，湿度条件类型为{data[12]}，最大干球温度下的湿球温度或露点温度为{data[13]}℃，湿度条件日计划名称为{data[14]}，最大干球湿度比为{data[15]}，最大干球温度下的焓为{data[16]}kJ/kg，日湿球温度范围为{data[17]}℃，气压为{data[18]}Pa，风速为{data[19]}m/s，风向为{data[20]}度，雨量指示器为{data[21]}，雪量指示器为{data[22]}，夏令时指示器为{data[23]}，太阳能模型指示器为{data[24]}，光束太阳能日计划名称为{data[25]}，漫射太阳日计划名称为{data[26]}，美国采暖、制冷与空调工程师学会（ASHRAE）光束辐照度晴空光学厚度为{data[27]}，美国采暖、制冷与空调工程师学会（ASHRAE）针对漫射辐照度的晴空光学厚度为{data[28]}，天空清澈度为{data[29]}，最大预热天数为{data[30]}，开始环境重置模式为{data[31]}。"
    return des

def _gen_description_all_materials(data:list):
    des = f"这是energyplus中的all_materials下的材料数据，在数据库中的id为{data[0]}，名称为{data[1]}，其材料类型为{data[2]}，如果是standard_material则在standard_materials表中的id为{data[3]}，如果是no_mass_material则在no_mass_materials表中的id为{data[4]}。"
    return des

def _update_description(db_path, table_name, data, gen_func):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    description = gen_func(data)
    sql = "UPDATE " + table_name + " SET description = ? WHERE id = ?"
    cursor.execute(sql, (description, data[0]))
    conn.commit()
    conn.close()

def update_description_material(db_path, data:list):
    _update_description(db_path, "standard_materials", data, _gen_description_material)

def update_description_nomass_material(db_path, data:list):
    _update_description(db_path, "no_mass_materials", data, _gen_description_nomass_material) # 误用

def update_description_construction(db_path, data:list):
    _update_description(db_path, "constructions", data, _gen_description_construction)

def update_description_schedule_type_limits(db_path, data:list):
    _update_description(db_path, "schedule_type_limits", data, _gen_description_schedule_type_limits)

def update_description_schedule_compact(db_path, data:list):
    _update_description(db_path, "schedule_compact", data, _gen_description_schedule_compact)

def update_description_sizingperiod_designday(db_path, data:list):
    _update_description(db_path, "sizingperiod_designday", data, _gen_description_sizingperiod_designday)

def update_description_all_materials(db_path, data:list):
    _update_description(db_path, "all_materials", data, _gen_description_all_materials)
