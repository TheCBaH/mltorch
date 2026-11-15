(* A contraction as a Machine IR helper: [contract.<descriptor>] taking the
   addresses of A, B and C (each its operand's first element), returning
   nothing, reading and writing memory, never failing. The descriptor is the
   helper's name ({!Contraction.name}), so a model, a realization and an
   artifact all recover it from the declaration alone. Each descriptor a
   program calls is its own helper. *)

let version = 1

(* Helper ids from 16 up, past the math helpers'. *)
let first_id = Mir_id.Helper.of_int 16

(* A contraction's position among the declared contractions, in lowering order. *)
module Index =
  Core.Tagged_int.Make
    (struct
      let prefix = "contraction"
    end)
    ()

let descriptor ~index (d : Contraction.t) =
  {
    Mir_helper.id =
      Mir_id.Helper.of_int (Mir_id.Helper.to_int first_id + Index.to_int index);
    name = Contraction.name d;
    version;
    params = [ Mir_type.Ptr; Mir_type.Ptr; Mir_type.Ptr ];
    results = [];
    effects = Mir_helper.Effect.Reads_writes;
    failures = [];
  }

(* The descriptor a helper declaration stands for, if it is a contraction. *)
let of_helper (h : Mir_helper.t) =
  if h.Mir_helper.version = version then Contraction.of_name h.Mir_helper.name
  else None
