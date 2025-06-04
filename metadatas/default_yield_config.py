DEFAULT_YIELD_CONFIG = {
    "dataset_dir":"datasets/buchwald_hartwig/test4",
    "dataset_name":"ReactionGraphBuchwaldHartwigTest4",
    "model_dir": "checkpoints/reaction_graph/buchwald_hartwig_test4",
    "seed": 333,
    "device": "cuda",
    "gpu": 7,
    "progress_bar": False,
    "loss_display_buffer": 100,
    "log_delta": 100,
    "network_config": {
        "graph_type": "reaction_graph",
        "dim_hidden": 1024,
        "dim_node_attribute": 36,
        "dim_edge_attribute": 10,
        "dim_edge_length": 0,
        "dim_hidden_features": 64,
        "message_passing_step": 3,
        "pooling_step": 3,
        "num_layers_pooling": 1,
        "dim_hidden_regression": 512,
        "aggregation_level": "molecule",
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
    }
}