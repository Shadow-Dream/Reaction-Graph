REACTION_GRAPH_USPTO_CONDITION_CONFIG = {
    "dataset_dir":"datasets/uspto_condition",
    "dataset_name":"ReactionGraphUSPTOCondition",
    "dim_catalyst": 54,
    "dim_solvent": 87,
    "dim_reagent": 235,
    "catalyst_none_index":20,
    "solvent_none_index":77,
    "reagent_none_index":200,
    "model_dir": "checkpoints/reaction_graph/uspto_condition",
    "seed": 123,
    "device": "cuda",
    "gpu": 0,
    "progress_bar": True,
    "loss_display_buffer": 100,
    "log_delta": 100,
    "checkpoint":"checkpoints/reaction_graph/uspto_condition/uspto_condition.ckpt",
    "network_config": {
        "graph_type": "reaction_graph",
        "dim_hidden": 4096,
        "dim_node_attribute": 110,
        "dim_edge_attribute": 13,
        "dim_edge_length": 16,
        "dim_hidden_features": 200,
        "message_passing_step": 3,
        "pooling_step": 2,
        "num_layers_pooling": 2
    },
    "training_config": {
        "pretrain_epoches":50,
        "finetune_epoches":50,
        "save_delta":10,
    },
    "pretrain_optimizer_config": {
        "lr":5e-4, 
        "weight_decay":1e-10
    },
    "pretrain_scheduler_config": {
        "mode":"min", 
        "factor":0.1, 
        "patience":5, 
        "min_lr":1e-7, 
        "verbose":True
    },
    "finetune_optimizer_config": {
        "lr":5e-4, 
        "weight_decay":1e-10
    },
    "finetune_scheduler_config": {
        "mode":"min", 
        "factor":0.1, 
        "patience":5, 
        "min_lr":1e-7, 
        "verbose":True
    },
    "criterion_config": {
        "max_gradient": 1e2,
        "none_weights": {
            "catalyst1":0.1,
            "solvent1":1,
            "solvent2":0.1,
            "reagent1":1,
            "reagent2":0.1
        },
        "smoothing": [0.9,0.8,0.8,0.7,0.7],
        "accumulation_steps": 4
    },
    "validate_config":{
        "validate_type":"val",
        "metric":[1,3,5,10,15],
        "beams":[1,3,1,5,1],
        "detail_level": "category"
    }
}