(* See arena_problem.mli. *)

module Kind = Alloc_script.Kind

module Kind_problem = struct
  type t = { kind : Kind.t; script : Tensor_id.t Interval_alloc.Script.t }
end

type t = {
  script : Alloc_script.t;
  eligible : Alloc_script.Alloc.t Tensor_id.Map.t;
  kinds : Kind_problem.t list;
}

let script t = t.script
let kinds t = t.kinds
let eligible t id = Tensor_id.Map.find_opt id t.eligible

let first_id t =
  match Tensor_id.Map.min_binding_opt t.eligible with
  | Some (id, _) -> id
  | None -> Tensor_id.of_int 0

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
  let events_of kind =
    List.filter_map
      (function
        | Alloc_script.Event.Alloc a
          when a.Alloc_script.Alloc.eligible
               && Kind.equal a.Alloc_script.Alloc.kind kind ->
            Some
              (Interval_alloc.Event.Alloc
                 {
                   key = a.Alloc_script.Alloc.id;
                   size = a.Alloc_script.Alloc.numel;
                 })
        | Alloc_script.Event.Free id -> (
            match Tensor_id.Map.find_opt id eligible with
            | Some a when Kind.equal a.Alloc_script.Alloc.kind kind ->
                Some (Interval_alloc.Event.Free id)
            | _ -> None)
        | Alloc_script.Event.Alloc _ | Alloc_script.Event.Node _ -> None)
      script
  in
  let problem kind =
    match events_of kind with
    | [] -> Err.return None
    | events ->
        let+ ia_script =
          Interval_alloc.Script.validate ~equal:Tensor_id.equal events
          |> Err.map_error ~pos:__POS__ (fun e ->
              match e with
              | `Double_alloc id
              | `Double_free id
              | `Free_unknown id
              | `Negative_size { Interval_alloc.Negative_size.key = id; _ } ->
                  `Arena_script id)
        in
        Some { Kind_problem.kind; script = ia_script }
  in
  let+ kinds = Err.List.map problem Kind.all in
  { script; eligible; kinds = List.filter_map Fun.id kinds }

let combined_bound_bytes t =
  Alloc_script.peak_bytes
    (List.filter
       (function
         | Alloc_script.Event.Alloc a -> a.Alloc_script.Alloc.eligible
         | Alloc_script.Event.Free id -> Tensor_id.Map.mem id t.eligible
         | Alloc_script.Event.Node _ -> false)
       t.script)

let out_of_arena_bytes t = Alloc_script.out_of_arena_bytes t.script
