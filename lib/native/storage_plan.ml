(* See storage_plan.mli. *)

open Core.Storage_units
module Arena_id = Storage_script.Arena_id

type t = {
  script : Storage_script.t;
  arenas : (Arena_id.t * Arena_plan.t) list;
}

let script t = t.script
let arenas t = t.arenas
let arena t id = List.assoc_opt id t.arenas

let lives_in (script : Storage_script.t) id =
  List.exists
    (function
      | Storage_script.Event.Alloc { Storage_script.Block.arena; _ } ->
          Option.equal Arena_id.equal arena (Some id)
      | _ -> false)
    (Storage_script.events script)

let create ?limits ?budget script =
  let open Err.Syntax in
  let+ arenas =
    Err.List.map
      (fun id ->
        let+ plan =
          Arena_plan.create ?limits ?budget
            ~alignment:(Storage_script.policy script)
            (Storage_script.arena_script script id)
        in
        (id, plan))
      (List.filter (lives_in script) Arena_id.all)
  in
  { script; arenas }

module Footprint = struct
  type t = {
    constants : Byte_size.t;
    execution : Byte_size.t;
    borrowed : Byte_size.t;
    outside : Byte_size.t;
  }

  let total t =
    let add a b =
      match Err.payload (Byte_size.add a b) with
      | Ok s -> s
      | Error (`Quantity_overflow _) ->
          Err.or_raise ~pp_error (Byte_size.of_int64 Int64.max_int)
    in
    List.fold_left add Byte_size.zero
      [ t.constants; t.execution; t.borrowed; t.outside ]

  let pp ppf t =
    Format.fprintf ppf
      "constant_bytes=%a execution_bytes=%a borrowed_bytes=%a outside_bytes=%a"
      Byte_size.pp t.constants Byte_size.pp t.execution Byte_size.pp t.borrowed
      Byte_size.pp t.outside
end

let footprint t =
  let open Err.Syntax in
  let first_id =
    List.find_map
      (function
        | Storage_script.Event.Alloc b ->
            Some b.Storage_script.Block.alloc.Alloc_script.Alloc.id
        | _ -> None)
      (Storage_script.events t.script)
    |> Option.value ~default:(Tensor_id.of_int 0)
  in
  let sum values =
    Err.List.fold_left
      (fun acc v ->
        Byte_size.add acc v
        |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
            `Peak_bytes_overflow first_id))
      Byte_size.zero values
  in
  let pools keep =
    sum
      (List.filter_map
         (fun (id, plan) ->
           if keep id then
             Some (Arena_plan.stats plan).Arena_plan.Stats.pool_bytes
           else None)
         t.arenas)
  in
  let is_constants = Arena_id.equal Arena_id.Constants in
  let* constants = pools is_constants in
  let* execution = pools (fun id -> not (is_constants id)) in
  let outside_of (b : Storage_script.Block.t) = Option.is_none b.arena in
  let* borrowed =
    Storage_script.peak_bytes t.script ~where:(fun b ->
        outside_of b
        &&
        match b.role with
        | Storage_script.Role.Constant | Storage_script.Role.Input -> true
        | Storage_script.Role.Intermediate | Storage_script.Role.Output -> false)
  in
  let+ outside =
    Storage_script.peak_bytes t.script ~where:(fun b ->
        outside_of b
        &&
        match b.role with
        | Storage_script.Role.Constant | Storage_script.Role.Input -> false
        | Storage_script.Role.Intermediate | Storage_script.Role.Output -> true)
  in
  { Footprint.constants; execution; borrowed; outside }
