(** [model_main.c]: the standalone application around [model_run]. The text does
    not depend on the model: everything model-specific is the constants and
    [model_run] of the inference unit, declared through {!C_model_abi}. *)

val source : string
