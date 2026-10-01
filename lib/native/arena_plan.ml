(* See arena_plan.mli. *)

module Kind = Alloc_script.Kind

module Slot = struct
  type t = {
    id : Tensor_id.t;
    signature : Tensor_sig.t;
    kind : Kind.t;
    offset : int64;
    numel : int64;
  }
end

module Pool = struct
  type t = { kind : Kind.t; numel : int64; bytes : int64 }
end

module Stats = struct
  type t = {
    kinds : (Kind.t * Interval_alloc.Stats.t) list;
    pool_bytes : int64;
    per_kind_bound_bytes : int64;
    combined_bound_bytes : int64;
    out_of_arena_bytes : int64;
  }

  let pp ppf t =
    Format.fprintf ppf
      "@[<v>(for this node order)@,\
       pool %Ld bytes, per-kind bound %Ld, combined bound %Ld, outside the \
       arena %Ld"
      t.pool_bytes t.per_kind_bound_bytes t.combined_bound_bytes
      t.out_of_arena_bytes;
    List.iter
      (fun (kind, stats) ->
        Format.fprintf ppf "@,%a: %a" Kind.pp kind Interval_alloc.Stats.pp stats)
      t.kinds;
    Format.fprintf ppf "@]"
end

module Over_limit = struct
  type limit = Bytes of int64 | Elements of int64
  type t = { kind : Kind.t; numel : int64; bytes : int64; limit : limit }
end

module Placement_error = struct
  type t =
    [ `Duplicate_placement of Tensor_id.t
    | `Live_overflow of Tensor_id.t
    | `Negative_offset of Tensor_id.t
    | `Offset_overflow of Tensor_id.t
    | `Out_of_pool of Tensor_id.t Interval_alloc.Out_of_pool.t
    | `Overlap of Tensor_id.t Interval_alloc.Overlap.t
    | `Pool_overflow of Tensor_id.t
    | `Unknown_key of Tensor_id.t
    | `Unplaced of Tensor_id.t ]
end

type error =
  [ `Arena_over_limit of Over_limit.t
  | `Arena_placement of Placement_error.t
  | `Arena_script of Tensor_id.t ]

type t = {
  script : Alloc_script.t;
  slots : Slot.t Tensor_id.Map.t;
  pools : Pool.t list;
  stats : Stats.t;
}

let script t = t.script
let slot t id = Tensor_id.Map.find_opt id t.slots
let slots t = List.map snd (Tensor_id.Map.bindings t.slots)
let pools t = t.pools
let stats t = t.stats

let add a b =
  if Int64.compare a (Int64.sub Int64.max_int b) > 0 then None
  else Some (Int64.add a b)

(* Sums stay far below [int64] (a pool is bounded by [Hard.numel] cells), but a
   sum of bounds is a different aggregate, so it is checked all the same. *)
let sum_checked id values =
  Err.List.fold_left
    (fun acc v ->
      match add acc v with
      | Some s -> Err.return s
      | None -> Err.fail ~pos:__POS__ (`Arena_placement (`Live_overflow id)))
    0L values

let create ?(limits = Kernel.Limits.default) ?budget (script : Alloc_script.t) =
  let open Err.Syntax in
  let* problem = Arena_problem.of_script script in
  let first_id = Arena_problem.first_id problem in
  let plan_kind { Arena_problem.Kind_problem.kind; script = ia_script } =
    let placement e = `Arena_placement (e : Placement_error.t) in
    let* solution, stats =
      Interval_alloc.solve_best ?budget ia_script
      |> Err.map_error ~pos:__POS__ (function
          | (`Live_overflow _ | `Pool_overflow _) as e -> placement e)
    in
    let* witness =
      Interval_alloc.check ia_script solution
      |> Err.map_error ~pos:__POS__ (fun e -> placement e)
    in
    let numel = Interval_alloc.pool witness in
    let bytes = Int64.mul numel (Kind.cell_bytes kind) in
    let over limit =
      Err.fail ~pos:__POS__
        (`Arena_over_limit { Over_limit.kind; numel; bytes; limit })
    in
    if Int64.compare numel Kernel.Limits.Hard.numel >= 0 then
      over (Over_limit.Elements Kernel.Limits.Hard.numel)
    else if Int64.compare bytes limits.Kernel.Limits.max_bytes > 0 then
      over (Over_limit.Bytes limits.Kernel.Limits.max_bytes)
    else Err.return (kind, witness, stats, numel, bytes)
  in
  let* planned = Err.List.map plan_kind (Arena_problem.kinds problem) in
  let slots =
    List.fold_left
      (fun acc (kind, witness, _, _, _) ->
        List.fold_left
          (fun acc (id, offset) ->
            let a =
              match Arena_problem.eligible problem id with
              | Some a -> a
              | None -> assert false
            in
            Tensor_id.Map.add id
              {
                Slot.id;
                signature = a.Alloc_script.Alloc.signature;
                kind;
                offset;
                numel = a.Alloc_script.Alloc.numel;
              }
              acc)
          acc
          (Interval_alloc.placements witness))
      Tensor_id.Map.empty planned
  in
  let pools =
    List.map
      (fun (kind, _, _, numel, bytes) -> { Pool.kind; numel; bytes })
      planned
  in
  let* pool_bytes =
    sum_checked first_id (List.map (fun (p : Pool.t) -> p.Pool.bytes) pools)
  in
  let* per_kind_bound_bytes =
    sum_checked first_id
      (List.map
         (fun (kind, _, (s : Interval_alloc.Stats.t), _, _) ->
           Int64.mul s.Interval_alloc.Stats.lower_bound (Kind.cell_bytes kind))
         planned)
  in
  let* combined_bound_bytes =
    Arena_problem.combined_bound_bytes problem
    |> Err.map_error ~pos:__POS__ (fun (`Peak_bytes_overflow id) ->
        `Arena_placement (`Live_overflow id))
  in
  let* out_of_arena_bytes =
    Arena_problem.out_of_arena_bytes problem
    |> Err.map_error ~pos:__POS__ (fun (`Peak_bytes_overflow id) ->
        `Arena_placement (`Live_overflow id))
  in
  Err.return
    {
      script;
      slots;
      pools;
      stats =
        {
          Stats.kinds = List.map (fun (k, _, s, _, _) -> (k, s)) planned;
          pool_bytes;
          per_kind_bound_bytes;
          combined_bound_bytes;
          out_of_arena_bytes;
        };
    }
