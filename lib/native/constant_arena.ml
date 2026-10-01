(* See constant_arena.mli. *)

open Core.Storage_units

module Generation = struct
  type t = int64

  let equal = Int64.equal
  let pp ppf t = Format.fprintf ppf "v%Ld" t
end

let next = ref 0L

type t = {
  script : Storage_script.t;
  generation : Generation.t;
  arena : Arena.t option;
      (* Held so the views stay the storage of this version: never read. *)
  bindings : (Tensor_id.t * Tensor.packed) list;
  loaded : Arena.Copies.t;
}

let create plan ~constants =
  let open Err.Syntax in
  let* arena =
    match Storage_plan.arena plan Storage_plan.Arena_id.Constants with
    | None -> Err.return None
    | Some p ->
        let+ a = Arena.create p in
        Some a
  in
  let blocks =
    List.filter_map
      (function
        | Storage_script.Event.Alloc
            ({ Storage_script.Block.role = Storage_script.Role.Constant; _ } as
             b) ->
            Some b
        | _ -> None)
      (Storage_script.events (Storage_plan.script plan))
  in
  let* bound =
    Err.List.map
      (fun (b : Storage_script.Block.t) ->
        let id = b.alloc.Alloc_script.Alloc.id in
        let* src =
          List.assoc_opt id constants
          |> Err.of_option ~pos:__POS__ (`Missing_constant id)
        in
        let* slot =
          match arena with None -> Err.return None | Some a -> Arena.view a id
        in
        match slot with
        | None -> Err.return ((id, src), None)
        | Some dst ->
            let+ () = Tensor.blit_into src dst in
            ((id, dst), Some b.alloc.Alloc_script.Alloc.bytes))
      blocks
  in
  let generation = !next in
  next := Int64.succ generation;
  (* A constant set is bounded by its arena, far below [int64]. *)
  let loaded =
    List.fold_left
      (fun (c : Arena.Copies.t) -> function
        | _, None -> c
        | _, Some bytes ->
            {
              Arena.Copies.count = Int64.succ c.count;
              bytes = Err.or_raise ~pp_error (Byte_size.add c.bytes bytes);
            })
      { Arena.Copies.count = 0L; bytes = Byte_size.zero }
      bound
  in
  Err.return
    {
      script = Storage_plan.script plan;
      generation;
      arena;
      bindings = List.map fst bound;
      loaded;
    }

let script t = t.script
let generation t = t.generation
let bindings t = t.bindings
let loaded t = t.loaded

let arena_bytes t =
  match t.arena with
  | None -> Byte_size.zero
  | Some a -> (Arena_plan.stats (Arena.plan a)).Arena_plan.Stats.pool_bytes
