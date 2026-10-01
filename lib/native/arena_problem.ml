(* See arena_problem.mli. *)

open Core.Storage_units
module Kind = Alloc_script.Kind

module Kind_problem = struct
  type t = {
    kind : Kind.t;
    script : Tensor_id.t Interval_alloc.Script.t;
    padded : Tensor_id.t Interval_alloc.Script.t;
  }
end

type t = {
  script : Alloc_script.t;
  eligible : Alloc_script.Alloc.t Tensor_id.Map.t;
  kinds : Kind_problem.t list;
  combined : Tensor_id.t Interval_alloc.Script.t option;
}

let script t = t.script
let kinds t = t.kinds
let eligible t id = Tensor_id.Map.find_opt id t.eligible

let first_id t =
  match Tensor_id.Map.min_binding_opt t.eligible with
  | Some (id, _) -> id
  | None -> Tensor_id.of_int 0

(* A slot's payload, exact, or padded to its alignment for a minimum. A payload
   is bounded far below [int64], so the padding cannot overflow. *)
let exact (a : Alloc_script.Alloc.t) = a.Alloc_script.Alloc.bytes

let padded (a : Alloc_script.Alloc.t) =
  Err.or_raise ~pp_error
    (Byte_alignment.pad a.Alloc_script.Alloc.bytes
       a.Alloc_script.Alloc.alignment)

let of_script (script : Alloc_script.t) =
  let open Err.Syntax in
  let eligible =
    List.fold_left
      (fun acc -> function
        | Alloc_script.Event.Alloc a when a.Alloc_script.Alloc.eligible ->
            Tensor_id.Map.add a.Alloc_script.Alloc.id a acc
        | _ -> acc)
      Tensor_id.Map.empty script
  in
  (* The eligible events of the kinds [keep] admits, each block [extent]. *)
  let events extent keep =
    List.filter_map
      (function
        | Alloc_script.Event.Alloc a
          when a.Alloc_script.Alloc.eligible && keep a.Alloc_script.Alloc.kind
          ->
            Some
              (Interval_alloc.Event.Alloc
                 {
                   key = a.Alloc_script.Alloc.id;
                   size = extent a;
                   alignment = a.Alloc_script.Alloc.alignment;
                 })
        | Alloc_script.Event.Free id -> (
            match Tensor_id.Map.find_opt id eligible with
            | Some a when keep a.Alloc_script.Alloc.kind ->
                Some (Interval_alloc.Event.Free id)
            | _ -> None)
        | Alloc_script.Event.Alloc _ | Alloc_script.Event.Node _ -> None)
      script
  in
  let problem extent keep =
    match events extent keep with
    | [] -> Err.return None
    | events ->
        let+ ia_script =
          Interval_alloc.Script.validate ~equal:Tensor_id.equal events
          |> Err.map_error ~pos:__POS__ (fun e ->
              match e with
              | `Double_alloc id | `Double_free id | `Free_unknown id ->
                  `Arena_script id)
        in
        Some ia_script
  in
  let kind_problem kind =
    let* script = problem exact (Kind.equal kind) in
    let+ padded = problem padded (Kind.equal kind) in
    match (script, padded) with
    | Some script, Some padded -> Some { Kind_problem.kind; script; padded }
    | _ -> None
  in
  let* kinds = Err.List.map kind_problem Kind.all in
  let+ combined = problem padded (fun _ -> true) in
  { script; eligible; kinds = List.filter_map Fun.id kinds; combined }

let combined_bound_bytes t =
  match t.combined with
  | None -> Err.return Byte_size.zero
  | Some s ->
      Interval_alloc.lower_bound s
      |> Err.map_error ~pos:__POS__ (fun (`Live_overflow id) ->
          `Peak_bytes_overflow id)

let out_of_arena_bytes t = Alloc_script.out_of_arena_bytes t.script
