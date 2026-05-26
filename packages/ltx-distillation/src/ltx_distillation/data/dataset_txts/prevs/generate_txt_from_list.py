

data_path_list = [
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/vfhq",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/hdtf_origin",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/avspeech/avspeech_chaoqingxi",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/avspeech/avspeech_qingxi",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/celebv_text/celebv_text_02",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/celebv_text/celebv_text_03",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/open_source_dataset/singinghead",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_20_1_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_20_1_2",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_36_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_36_2",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_36_3",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_36_5",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_36_6",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_36_7",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_1_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_1_2",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_2_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_2_2",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_3_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_3_2",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_4_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_5_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_6_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_55_7",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_66_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_66_2",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_71_1",
                    "/gemini-1/space/human_guozz2/data/wzh/omnihuman_dataset/general_data_1/pack_71_2",
                    ]

with open("variance_done.txt", "w") as f:
    for i in data_path_list:
        f.write(i+"/config_hys_gather.json"+"\n")
