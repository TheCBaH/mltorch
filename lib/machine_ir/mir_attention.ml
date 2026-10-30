(* An attention as a Machine IR helper: [attend.<descriptor>] taking the
   addresses of Q, K, V, O and, when the descriptor has a mask, the mask (each
   operand's first element), returning nothing, reading and writing memory,
   never failing. The descriptor is the helper's name
   ({!Attention_block.name}), so a model, a realization and an artifact all
   recover it from the declaration alone. *)

let version = 1

(* Helper ids from 1024 up, past the contractions'. *)
let first_id = 1024

let descriptor ~index (d : Attention_block.t) =
  {
    Mir_helper.id = Mir_id.Helper.of_int (first_id + index);
    name = Attention_block.name d;
    version;
    params =
      List.init
        (if d.Attention_block.mask = None then 4 else 5)
        (fun _ -> Mir_type.Ptr);
    results = [];
    effects = Mir_helper.Effect.Reads_writes;
    failures = [];
  }

(* The descriptor a helper declaration stands for, if it is an attention. *)
let of_helper (h : Mir_helper.t) =
  if h.Mir_helper.version = version then
    Attention_block.of_name h.Mir_helper.name
  else None
