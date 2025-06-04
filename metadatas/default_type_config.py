DEFAULT_TYPE_CONFIG = {
    "dataset_dir":"datasets/uspto_tpl",
    "dataset_name":"ReactionGraphUSPTOTPL",
    "num_types": 1000,
    "model_dir": "checkpoints/reaction_graph/uspto_tpl",
    "seed": 333,
    "device": "cuda",
    "gpu": 7,
    "progress_bar": True,
    "loss_display_buffer": 100,
    "log_delta": 100,
    "network_config": {
        "graph_type": "reaction_graph",
        "dim_hidden": 4096,
        "dim_node_attribute": 135,
        "dim_edge_attribute": 14,
        "dim_edge_length": 16,
        "dim_hidden_features": 200,
        "message_passing_step": 3,
        "pooling_step": 2,
        "num_layers_pooling": 2
    },
    "training_config": {
        "epoches":100,
        "save_delta":10,
    },
    "optimizer_config": {
        "lr":5e-4, 
        "weight_decay":1e-10
    },
    "scheduler_config": {
        "mode":"min", 
        "factor":0.1, 
        "patience":5, 
        "min_lr":1e-7, 
        "verbose":True
    },
    "criterion_config": {
        "max_gradient": 1e2,
        "accumulation_steps": 4
    },
    "validate_config":{
        "validate_type":"val",
        "detail_level": "results"
    }
}