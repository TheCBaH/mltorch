(* See storage_run.mli. *)

open Graph_ir
open Core.Storage_units
module S = Storage_script

type error = Arena_run.error

let pp_error = Arena_run.pp_error

(* A tally: no runner copies 2^63 bytes. *)
let tally (c : Arena.Copies.t) bytes =
  {
    Arena.Copies.count = Int64.succ c.count;
    bytes =
      Err.or_raise ~pp_error:Core.Storage_units.pp_error
        (Byte_size.add c.bytes bytes);
  }

let no_copies = { Arena.Copies.count = 0L; bytes = Byte_size.zero }

type t = {
  plan : Storage_plan.t;
  constants : Constant_arena.t;
  scratch : (S.Arena_id.t * Arena.t) list;
  results_id : S.Arena_id.t;
  results_plan : Arena_plan.t option;
  max_outstanding : int64;
  poison : Arena.Poison.t option;
  mutable results : Arena.t list;
  mutable generation : Result_lease.Generation.t;
  mutable input_copies : Arena.Copies.t;
}

let plan t = t.plan

let create ?poison ?(max_outstanding = 1L) plan constants =
  let open Err.Syntax in
  let script = Storage_plan.script plan in
  let* () =
    match S.first_difference script (Constant_arena.script constants) with
    | Some position -> Err.fail ~pos:__POS__ (`Storage_script_mismatch position)
    | None -> Err.return ()
  in
  let results_id =
    match (S.config script).S.Config.layout with
    | S.Layout.Separate -> S.Arena_id.Outputs
    | S.Layout.Shared_execution -> S.Arena_id.Execution
  in
  let+ scratch =
    Err.List.map
      (fun (id, p) ->
        let+ a = Arena.create ?poison p in
        (id, a))
      (List.filter
         (fun (id, _) ->
           not
             (S.Arena_id.equal id S.Arena_id.Constants
             || S.Arena_id.equal id results_id))
         (Storage_plan.arenas plan))
  in
  {
    plan;
    constants;
    scratch;
    results_id;
    results_plan = Storage_plan.arena plan results_id;
    max_outstanding = Int64.max 1L max_outstanding;
    poison;
    results = [];
    generation = Result_lease.first_generation;
    input_copies = no_copies;
  }

let outstanding t =
  Int64.of_int (List.length (List.filter Arena.busy t.results))

(* A free result arena, pinned; a new one while under the bound. *)
let acquire_results t =
  let open Err.Syntax in
  match t.results_plan with
  | None -> Err.return None
  | Some p -> (
      match List.find_opt (fun a -> not (Arena.busy a)) t.results with
      | Some a ->
          let+ () = Arena.acquire a in
          Some a
      | None ->
          if
            Int64.compare
              (Int64.of_int (List.length t.results))
              t.max_outstanding
            >= 0
          then Err.fail ~pos:__POS__ `Arena_busy
          else
            let* a = Arena.create ?poison:t.poison p in
            t.results <- t.results @ [ a ];
            let+ () = Arena.acquire a in
            Some a)

(* Acquire every scratch arena, or none. *)
let acquire_scratch t =
  let rec go held = function
    | [] -> Err.return ()
    | (_, a) :: rest -> (
        match Arena.acquire a with
        | Ok () -> go (a :: held) rest
        | Error _ as e ->
            List.iter Arena.release held;
            e)
  in
  go [] t.scratch

(* The caller's inputs, each copied into its slot when the script places it. *)
let bind_inputs t ~arena_of (g : graph) inputs =
  let open Err.Syntax in
  let blocks =
    List.filter_map
      (function
        | S.Event.Alloc ({ S.Block.role = S.Role.Input; _ } as b) -> Some b
        | _ -> None)
      (S.events (Storage_plan.script t.plan))
  in
  Err.List.map
    (fun (id, src) ->
      match
        List.find_opt
          (fun (b : S.Block.t) ->
            Tensor_id.equal b.alloc.Alloc_script.Alloc.id id)
          blocks
      with
      | Some { S.Block.arena = Some aid; alloc; _ } -> (
          let* slot =
            match arena_of aid with
            | None -> Err.return None
            | Some a -> Arena.view a id
          in
          match slot with
          | None -> Err.fail ~pos:__POS__ (`Arena_slot id)
          | Some dst ->
              let+ () = Tensor.blit_into src dst in
              t.input_copies <-
                tally t.input_copies alloc.Alloc_script.Alloc.bytes;
              (id, dst))
      | _ -> Err.return (id, src))
    (List.filter (fun (id, _) -> List.mem id g.Graph.inputs) inputs)

let run t ?hooks ?region_executor ?region_group_executor ?node_executor ?limits
    ?retain (g : graph) ~inputs =
  let open Err.Syntax in
  let* results = acquire_results t |> Err.map_error (fun e -> (e :> error)) in
  let release_results () = Option.iter Arena.release results in
  match acquire_scratch t with
  | Error e ->
      release_results ();
      Error (e :> error Err.Error.t)
  | Ok () -> (
      let arena_of id =
        if S.Arena_id.equal id t.results_id then results
        else List.assoc_opt id t.scratch
      in
      let body () =
        let* inputs =
          bind_inputs t ~arena_of g inputs
          |> Err.map_error (fun e -> (e :> error))
        in
        Eval_direct.run_storage
          ~arenas:(List.map snd t.scratch @ Option.to_list results)
          ~script:(Storage_plan.script t.plan)
          ?hooks ?region_executor ?region_group_executor ?node_executor ?limits
          ?retain
          ~constants:(Constant_arena.bindings t.constants)
          g ~inputs
        |> Err.map_error (fun e -> (e :> error))
      in
      match
        Fun.protect
          ~finally:(fun () ->
            List.iter (fun (_, a) -> Arena.release a) t.scratch)
          body
      with
      | exception exn ->
          release_results ();
          raise exn
      | Error _ as e ->
          release_results ();
          e
      | Ok outputs ->
          let generation = t.generation in
          t.generation <- Result_lease.next_generation generation;
          Err.return
            (Result_lease.make ~arena:results ~constants:t.constants ~generation
               ~signatures:
                 (Tensor_id.Map.filter
                    (fun id _ -> Tensor_id.Map.mem id outputs)
                    g.Graph.tensors)
               outputs))

module Report = struct
  type t = {
    footprint : Storage_plan.Footprint.t;
    constant_generation : Constant_arena.Generation.t;
    result_arenas : int64;
    result_arena_bytes : Byte_size.t;
    outstanding : int64;
    constant_loads : Arena.Copies.t;
    input_copies : Arena.Copies.t;
    mixed_mode_copies : Arena.Copies.t;
  }

  let pp_copies ppf (c : Arena.Copies.t) =
    Format.fprintf ppf "%Ld (%a bytes)" c.count Byte_size.pp c.bytes

  let pp ppf t =
    Format.fprintf ppf
      "%a constants=%a result_arenas=%Ld result_arena_bytes=%a outstanding=%Ld \
       constant_loads=%a input_copies=%a mixed_mode_copies=%a"
      Storage_plan.Footprint.pp t.footprint Constant_arena.Generation.pp
      t.constant_generation t.result_arenas Byte_size.pp t.result_arena_bytes
      t.outstanding pp_copies t.constant_loads pp_copies t.input_copies
      pp_copies t.mixed_mode_copies
end

let report t =
  let open Err.Syntax in
  let+ footprint = Storage_plan.footprint t.plan in
  let arenas = List.map snd t.scratch @ t.results in
  {
    Report.footprint;
    constant_generation = Constant_arena.generation t.constants;
    result_arenas = Int64.of_int (List.length t.results);
    result_arena_bytes =
      Option.fold ~none:Byte_size.zero
        ~some:(fun p -> (Arena_plan.stats p).Arena_plan.Stats.pool_bytes)
        t.results_plan;
    outstanding = outstanding t;
    constant_loads = Constant_arena.loaded t.constants;
    input_copies = t.input_copies;
    mixed_mode_copies =
      List.fold_left
        (fun acc a ->
          let c = Arena.copies a in
          {
            Arena.Copies.count = Int64.add acc.Arena.Copies.count c.count;
            bytes =
              Err.or_raise ~pp_error:Core.Storage_units.pp_error
                (Byte_size.add acc.bytes c.bytes);
          })
        no_copies arenas;
  }
