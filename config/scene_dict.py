left_0_set = {
    "lego_entity","matrixcity","metal_materials","desk_bookshelf","orchids","china_art_museum","traffic","ficus","shoe_rack"
}
left_1_set = {
    "chair","lego_bulldozer","wukang_mansion","truck","room_floor","hotdog","street"
}
scene_dict = {
    
    "chair":{
        # 1500
        # 0 right; 1 left 
        "scale_physical2world":0.055,
        "physical_width":78.55,
        "thickness":9.09,
        "vertical":"z",
        "orientation":"xoz"
    },
    "lego_entity":{
        # 0 left; 1 right
        "scale_physical2world":0.07716,
        "physical_width":77.76,
        "thickness":11,
        "ground":-0.5,
        "vertical":"x",
        "orientation":"yox"
    },
    "matrixcity":{
        # 0 left; 1 right
        "scale_physical2world":0.23,
        "thickness":5,
        "vertical":"y",
        "delta_y":1.5,
        "orientation":"xoy"
    },
    "lego_bulldozer":{
        # 1 left; 0 right
        "scale_physical2world":0.083,
        "thickness":6,
        "vertical":"z",
        "ground_coefficient":-0.5,
        "orientation":"yoz"
    },
    # "minecraft":{
    #     "scale_physical2world":2,
    #     "thickness":74.07,
    #     "vertical":"y",
    #     "orientation":"zox"
    # },
    "minecraft":{
        "scale_physical2world":2,
        "thickness":4,
        "vertical":"y",
        "orientation":"zox"
    },
    "metal_materials":{
        # 0 left; 1 right
        "scale_physical2world":0.025,
        "thickness":12,
        "physical_width":103.68,
        "vertical":"y",
        "orientation":"xoy"
    },
    "wukang_mansion":{
        # 1 left; 0 right
        "scale_physical2world":0.0667,
        "thickness":6,
        "vertical":"z",
        "ground_coefficient":-1,
        "orientation":"yoz"
    },
    "truck":{
        # 1 left; 0 right
        "scale_physical2world":0.083,
        "thickness":6,
        "physical_width":52.05,
        "vertical":"z",
        "ground_coefficient":-0.5,
        "orientation":"yoz"
    },
    "room_floor":{
        # 1 left; 0 right
        "scale_physical2world":0.015,
        "thickness":46.7,
        "physical_width":288,
        "vertical":"z",
        "ground_coefficient":-0.5,
        "delta_y":0.35,
        "delta_z":0.15,
        "orientation":"yoz"
    },
    "desk_bookshelf":{
        # 0 left; 1 right
        "scale_physical2world":0.025,
        "thickness":6,
        "physical_width":172.8,
        "vertical":"z",
        "orientation":"yoz"
    },
    # "hotdog":{
    #     # 1 left; 0 right
    #     "scale_physical2world":0.06,
    #     "thickness":10,
    #     "physical_width":43.2,
    #     "vertical":"z",
    #     "orientation":"xoz",
    #     "ground_coefficient":-0.5,
    # },
    "hotdog":{
        # 1 left; 0 right
        "scale_physical2world":0.06,
        "thickness":2.17,
        "physical_width":57.6,
        "vertical":"-x",
        "orientation":"yox",
        "ground_coefficient":-0.5,
    },
    "orchids":{
        # 0 left; 1 right
        "scale_physical2world":0.04,
        "thickness":100,
        "physical_width":432,
        "vertical":"z",
        "orientation":"yoz",
        "ground_coefficient":-1.2,
    },
    "china_art_museum":{
        # 1 right; 0 left
        "scale_physical2world":0.03,
        "thickness":25,
        "physical_width":115.2,
        "vertical":"z",
        "orientation":"yoz",
    },
    "traffic":{
        # 0 left; 1 right
        "scale_physical2world":0.04,
        "thickness":8.75,
        "physical_width":108,
        "vertical":"z",
        "orientation":"yoz",
    },
    "ficus":{
        # 0 left; 1 right
        "scale_physical2world":0.05,
        "thickness":10,
        "physical_width":77.76,
        "vertical":"z",
        "orientation":"xoz",
        "ground_coefficient":-0.5,
    },
    "shoe_rack":{
        # 0 left; 1 right
        "scale_physical2world":0.083,
        "thickness":12,
        "physical_width":104.1,
        "vertical":"z",
        "orientation":"yoz",
        "ground_coefficient":-0.5,
    },
    "street":{
        # 1 left; 0 right
        "scale_physical2world":0.02,
        "thickness":100,
        "physical_width":648,
        "vertical":"z",
        "orientation":"yoz",
        "ground":-1,
    },
}
scene_id2key = list(scene_dict.keys())

object_dict = {
    "scale_physical2world":0.21,
    "thickness":6,
    "vertical":"y",
    "ground_coefficient":-0.5,
    "orientation":"xoy"
}

if __name__ == '__main__':
    
    for i in range(len(scene_id2key)):
        thickness = 6
        key = scene_id2key[i]
        if scene_dict[key].get("thickness") != None:
            thickness = scene_dict[key].get("thickness")
        print("{}: {}".format(key, 1/(thickness/100/2)))
    print("real machine: {}".format(1/(6/100/2)))