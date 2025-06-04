REACTION_GRAPH_HTE_CONFIG = {
    "device": "cuda",
    "progress_bar": False,
    "loss_display_buffer": 100,
    "log_delta": 100,
    "network_config": {
        "graph_type": "reaction_graph",
        "dim_hidden": 1024,
        "dim_edge_length": 0,
        "dim_hidden_features": 64,
        "message_passing_step": 3,
        "pooling_step": 3,
        "num_layers_pooling": 1,
        "dim_hidden_regression": 512,
        "aggregation_level": "molecule",
        "aggregation_method": "sumplus",
        "dropout":0.1
    },
    "training_config": {
        "epoches":500,
        "save_delta":50,
    },
    "optimizer_config": {
        "lr":1e-3, 
        "weight_decay":1e-5
    },
    "scheduler_config": {
        "milestones":[400, 450], 
        "gamma":0.1,
        "verbose":True
    },
    "criterion_config": {
        "kl":0.1,
        "max_gradient": 1e2,
        "accumulation_steps": 4
    },
    "validate_config":{
        "validate_type":"val",
        "detail_level": "results",
        "num_inference_pass": 30
    },
    "experiments":{
        "buchwald_hartwig_split1":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split1",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit1",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split1",
            "gpu": "0",
            "seed": "2",
            "mean":32.70961316731369,
            "std":27.15759042359038,
            "var":737.5347176154879,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split1/buchwald_hartwig_split1.ckpt"
        },
        "buchwald_hartwig_split2":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split2",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit2",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split2",
            "gpu": "1",
            "seed": "4",
            "mean":33.551046104338276,
            "std":27.502790324659102,
            "var":756.4034756421623,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split2/buchwald_hartwig_split2.ckpt"
        },
        "buchwald_hartwig_split3":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split3",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit3",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split3",
            "gpu": "2",
            "seed": "8",
            "mean":33.202214787685215,
            "std":27.373134597638185,
            "var":749.2884977004165,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split3/buchwald_hartwig_split3.ckpt"
        },
        "buchwald_hartwig_split4":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split4",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit4",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split4",
            "gpu": "3",
            "seed": "16",
            "mean":32.88250590525298,
            "std":27.34613960211944,
            "var":747.8113511386051,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split4/buchwald_hartwig_split4.ckpt"
        },
        "buchwald_hartwig_split5":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split5",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit5",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split5",
            "gpu": "4",
            "seed": "32",
            "mean":32.844330778073726,
            "std":27.080833753729085,
            "var":733.3715567971126,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split5/buchwald_hartwig_split5.ckpt"
        },
        "buchwald_hartwig_split6":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split6",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit6",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split6",
            "gpu": "5",
            "seed": "64",
            "mean":33.08662738098988,
            "std":27.3517377058204,
            "var":748.1175555279975,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split6/buchwald_hartwig_split6.ckpt"
        },
        "buchwald_hartwig_split7":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split7",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit7",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split7",
            "gpu": "6",
            "seed": "128",
            "mean":32.84735688513987,
            "std":27.30619887547046,
            "var":745.6284970267443,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split7/buchwald_hartwig_split7.ckpt"
        },
        "buchwald_hartwig_split8":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split8",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit8",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split8",
            "gpu": "7",
            "seed": "256",
            "mean":33.44197671019154,
            "std":27.41417154571875,
            "var":751.5368015380956,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split8/buchwald_hartwig_split8.ckpt"
        },
        "buchwald_hartwig_split9":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split9",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit9",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split9",
            "gpu": "0",
            "seed": "512",
            "mean":33.14484676973184,
            "std":27.221046129585947,
            "var":740.985352389046,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split9/buchwald_hartwig_split9.ckpt"
        },
        "buchwald_hartwig_split10":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/split10",
            "dataset_name":"ReactionGraphBuchwaldHartwigSplit10",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/split10",
            "gpu": "1",
            "seed": "2048",
            "mean":33.310009919957714,
            "std":27.17545936697554,
            "var":738.5055918061388,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/split10/buchwald_hartwig_split10.ckpt"
        },
        "buchwald_hartwig_test1_1":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test1",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest1",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1",
            "gpu": "2",
            "seed": "4096",
            "mean":32.213843362500164,
            "std":27.233682282587996,
            "var":741.6734506689473,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1/buchwald_hartwig_test1.ckpt"
        },
        "buchwald_hartwig_test1_2":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test1",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest1",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1",
            "gpu": "3",
            "seed": "8192",
            "mean":32.213843362500164,
            "std":27.233682282587996,
            "var":741.6734506689473,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1/buchwald_hartwig_test1.ckpt"
        },
        "buchwald_hartwig_test1_3":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test1",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest1",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1",
            "gpu": "4",
            "seed": "16384",
            "mean":32.213843362500164,
            "std":27.233682282587996,
            "var":741.6734506689473,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1/buchwald_hartwig_test1.ckpt"
        },
        "buchwald_hartwig_test1_4":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test1",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest1",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1",
            "gpu": "5",
            "seed": "32768",
            "mean":32.213843362500164,
            "std":27.233682282587996,
            "var":741.6734506689473,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test1/buchwald_hartwig_test1.ckpt"
        },
        "buchwald_hartwig_test2_1":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test2",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest2",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2",
            "gpu": "6",
            "seed": "000000",
            "mean":32.767170297335184,
            "std":27.3373354549761,
            "var":747.3299097778934,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2/buchwald_hartwig_test2.ckpt"
        },
        "buchwald_hartwig_test2_2":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test2",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest2",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2",
            "gpu": "7",
            "seed": "010101",
            "mean":32.767170297335184,
            "std":27.3373354549761,
            "var":747.3299097778934,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2/buchwald_hartwig_test2.ckpt"
        },
        "buchwald_hartwig_test2_3":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test2",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest2",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2",
            "gpu": "0",
            "seed": "020202",
            "mean":32.767170297335184,
            "std":27.3373354549761,
            "var":747.3299097778934,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2/buchwald_hartwig_test2.ckpt"
        },
        "buchwald_hartwig_test2_4":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test2",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest2",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2",
            "gpu": "1",
            "seed": "030303",
            "mean":32.767170297335184,
            "std":27.3373354549761,
            "var":747.3299097778934,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test2/buchwald_hartwig_test2.ckpt"
        },
        "buchwald_hartwig_test3_1":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test3",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest3",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3",
            "gpu": "2",
            "seed": "040404",
            "mean":32.98280692358633,
            "std":27.04533813083481,
            "var":731.4503146111874,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3/buchwald_hartwig_test3.ckpt"
        },
        "buchwald_hartwig_test3_2":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test3",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest3",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3",
            "gpu": "3",
            "seed": "050505",
            "mean":32.98280692358633,
            "std":27.04533813083481,
            "var":731.4503146111874,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3/buchwald_hartwig_test3.ckpt"
        },
        "buchwald_hartwig_test3_3":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test3",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest3",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3",
            "gpu": "4",
            "seed": "060606",
            "mean":32.98280692358633,
            "std":27.04533813083481,
            "var":731.4503146111874,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3/buchwald_hartwig_test3.ckpt"
        },
        "buchwald_hartwig_test3_4":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test3",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest3",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3",
            "gpu": "5",
            "seed": "070707",
            "mean":32.98280692358633,
            "std":27.04533813083481,
            "var":731.4503146111874,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test3/buchwald_hartwig_test3.ckpt"
        },
        "buchwald_hartwig_test4_1":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test4",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest4",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4",
            "gpu": "6",
            "seed": "080808",
            "mean":33.75965661125696,
            "std":27.493774740024794,
            "var":755.9076494552254,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4/buchwald_hartwig_test4.ckpt"
        },
        "buchwald_hartwig_test4_2":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test4",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest4",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4",
            "gpu": "7",
            "seed": "090909",
            "mean":33.75965661125696,
            "std":27.493774740024794,
            "var":755.9076494552254,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4/buchwald_hartwig_test4.ckpt"
        },
        "buchwald_hartwig_test4_3":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test4",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest4",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4",
            "gpu": "0",
            "seed": "101010",
            "mean":33.75965661125696,
            "std":27.493774740024794,
            "var":755.9076494552254,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4/buchwald_hartwig_test4.ckpt"
        },
        "buchwald_hartwig_test4_4":{
            "dataset_dir":"datasets/hte/buchwald_hartwig/test4",
            "dataset_name":"ReactionGraphBuchwaldHartwigTest4",
            "model_dir": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4",
            "gpu": "1",
            "seed": "111111",
            "mean":33.75965661125696,
            "std":27.493774740024794,
            "var":755.9076494552254,
            "network_config":{
                "dim_node_attribute": 36,
                "dim_edge_attribute": 10
            },
            "checkpoint": "checkpoints/reaction_graph/hte/buchwald_hartwig/test4/buchwald_hartwig_test4.ckpt"
        },
        "suzuki_miyaura_split1":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split1",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit1",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split1",
            "gpu": "2",
            "seed": "121212",
            "mean":39.85437856611263,
            "std":28.155007028829313,
            "var":792.704420793428,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split1/suzuki_miyaura_split1.ckpt"
        },
        "suzuki_miyaura_split2":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split2",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit2",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split2",
            "gpu": "3",
            "seed": "131313",
            "mean":39.54031257752419,
            "std":28.016993457594904,
            "var":784.9519224029156,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split2/suzuki_miyaura_split2.ckpt"
        },
        "suzuki_miyaura_split3":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split3",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit3",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split3",
            "gpu": "4",
            "seed": "141414",
            "mean":39.782684197469614,
            "std":28.02892873943939,
            "var":785.6208462805714,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split3/suzuki_miyaura_split3.ckpt"
        },
        "suzuki_miyaura_split4":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split4",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit4",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split4",
            "gpu": "5",
            "seed": "151515",
            "mean":39.663607045398166,
            "std":28.081754389355964,
            "var":788.5849295841131,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split4/suzuki_miyaura_split4.ckpt"
        },
        "suzuki_miyaura_split5":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split5",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit5",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split5",
            "gpu": "6",
            "seed": "161616",
            "mean":39.96874224758125,
            "std":28.106215159086965,
            "var":789.95933056889,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split5/suzuki_miyaura_split5.ckpt"
        },
        "suzuki_miyaura_split6":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split6",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit6",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split6",
            "gpu": "7",
            "seed": "171717",
            "mean":39.57305879434384,
            "std":28.102411474524203,
            "var":789.7455306834696,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split6/suzuki_miyaura_split6.ckpt"
        },
        "suzuki_miyaura_split7":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split7",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit7",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split7",
            "gpu": "0",
            "seed": "181818",
            "mean":39.583229967749936,
            "std":27.81452295136681,
            "var":773.6476870121111,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split7/suzuki_miyaura_split7.ckpt"
        },
        "suzuki_miyaura_split8":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split8",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit8",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split8",
            "gpu": "1",
            "seed": "191919",
            "mean":39.67873976680725,
            "std":28.16047577600631,
            "var":793.0123959310382,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split8/suzuki_miyaura_split8.ckpt"
        },
        "suzuki_miyaura_split9":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split9",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit9",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split9",
            "gpu": "2",
            "seed": "202020",
            "mean":39.269908211361944,
            "std":27.919100000608346,
            "var":779.4761448439689,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split9/suzuki_miyaura_split9.ckpt"
        },
        "suzuki_miyaura_split10":{
            "dataset_dir":"datasets/hte/suzuki_miyaura/split10",
            "dataset_name":"ReactionGraphSuzukiMiyauraSplit10",
            "model_dir": "checkpoints/reaction_graph/hte/suzuki_miyaura/split10",
            "gpu": "3",
            "seed": "212121",
            "mean":39.61225502356735,
            "std":28.049310491575365,
            "var":786.7638190527998,
            "network_config":{
                "dim_node_attribute": 47,
                "dim_edge_attribute": 11
            },
            "checkpoint": "checkpoints/reaction_graph/hte/suzuki_miyaura/split10/suzuki_miyaura_split10.ckpt"
        }
    }
}