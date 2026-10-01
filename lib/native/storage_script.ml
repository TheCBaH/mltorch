(* See storage_script.mli. *)

open Graph_common
open Core.Storage_units

module Arena_id = struct
  type t = Constants | Execution | Inputs | Intermediates | Outputs

  let all = [ Constants; Execution; Inputs; Intermediates; Outputs ]
  let equal (a : t) b = a = b
  let compare (a : t) b = Stdlib.compare a b

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with
      | Constants -> "constants"
      | Execution -> "execution"
      | Inputs -> "inputs"
      | Intermediates -> "intermediates"
      | Outputs -> "outputs")
end

module Boundary = struct
  type t = Input_population | Model_init | Result_publication | Result_release

  let equal (a : t) b = a = b

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with
      | Input_population -> "input population"
      | Model_init -> "model init"
      | Result_publication -> "result publication"
      | Result_release -> "result release")
end

module Layout = struct
  type t = Separate | Shared_execution

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with
      | Separate -> "separate"
      | Shared_execution -> "shared_execution")
end

module Ownership = struct
  type t = Borrowed | Copied

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with Borrowed -> "borrowed" | Copied -> "copied")
end

module Config = struct
  type t = { layout : Layout.t; constants : Ownership.t; inputs : Ownership.t }

  let equal (a : t) b = a = b

  let pp ppf t =
    Format.fprintf ppf "layout=%a constants=%a inputs=%a" Layout.pp t.layout
      Ownership.pp t.constants Ownership.pp t.inputs
end

module Role = struct
  type t = Constant | Input | Intermediate | Output

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with
      | Constant -> "constant"
      | Input -> "input"
      | Intermediate -> "intermediate"
      | Output -> "output")
end

module Block = struct
  type t = {
    alloc : Alloc_script.Alloc.t;
    role : Role.t;
    arena : Arena_id.t option;
  }

  let equal a b =
    Alloc_script.Alloc.equal a.alloc b.alloc
    && a.role = b.role
    && Option.equal Arena_id.equal a.arena b.arena
end

module Event = struct
  type t =
    | Alloc of Block.t
    | Boundary of Boundary.t
    | Free of Tensor_id.t
    | Node of Node_id.t

  let equal a b =
    match (a, b) with
    | Alloc a, Alloc b -> Block.equal a b
    | Boundary a, Boundary b -> Boundary.equal a b
    | Free a, Free b -> Tensor_id.equal a b
    | Node a, Node b -> Node_id.equal a b
    | (Alloc _ | Boundary _ | Free _ | Node _), _ -> false

  let pp ppf = function
    | Alloc { Block.alloc = a; role; arena } ->
        Format.fprintf ppf "alloc %a %a %a %a bytes align %a in %a" Role.pp role
          Tensor_id.pp a.Alloc_script.Alloc.id Alloc_script.Kind.pp
          a.Alloc_script.Alloc.kind Byte_size.pp a.Alloc_script.Alloc.bytes
          Byte_alignment.pp a.Alloc_script.Alloc.alignment
          (Fmt.option ~none:(Fmt.any "no arena") Arena_id.pp)
          arena
    | Boundary b -> Format.fprintf ppf "-- %a" Boundary.pp b
    | Free id -> Format.fprintf ppf "free %a" Tensor_id.pp id
    | Node id -> Format.fprintf ppf "node %a" Node_id.pp id
end

type t = {
  config : Config.t;
  policy : Alignment_policy.t;
  events : Event.t list;
}

let make config policy events = { config; policy; events }
let config t = t.config
let policy t = t.policy
let events t = t.events

let arena_of (config : Config.t) (role : Role.t) ~quantized ~kept =
  let execution separate =
    match config.Config.layout with
    | Layout.Separate -> Some separate
    | Layout.Shared_execution -> Some Arena_id.Execution
  in
  if quantized then None
  else
    match role with
    | Role.Constant -> (
        match config.Config.constants with
        | Ownership.Borrowed -> None
        | Ownership.Copied -> Some Arena_id.Constants)
    | Role.Input -> (
        match config.Config.inputs with
        | Ownership.Borrowed -> None
        | Ownership.Copied ->
            execution (if kept then Arena_id.Outputs else Arena_id.Inputs))
    | Role.Intermediate -> execution Arena_id.Intermediates
    | Role.Output -> execution Arena_id.Outputs

let arena_script t id =
  List.filter_map
    (function
      | Event.Alloc { Block.alloc; arena; _ } ->
          Some
            (Alloc_script.Event.Alloc
               {
                 alloc with
                 Alloc_script.Alloc.eligible =
                   Option.equal Arena_id.equal arena (Some id);
               })
      | Event.Free i -> Some (Alloc_script.Event.Free i)
      | Event.Node n -> Some (Alloc_script.Event.Node n)
      | Event.Boundary _ -> None)
    t.events

let first_difference a b =
  if
    not
      (Config.equal a.config b.config
      && Alignment_policy.equal a.policy b.policy)
  then Some (Alloc_script.Position.of_int 0)
  else
    let rec go i a b =
      match (a, b) with
      | [], [] -> None
      | x :: a', y :: b' when Event.equal x y -> go (i + 1) a' b'
      | _ -> Some (Alloc_script.Position.of_int i)
    in
    go 0 a.events b.events

let peak_bytes t ~where =
  let open Err.Syntax in
  let+ _, _, peak =
    Err.List.fold_left
      (fun (resident, live, peak) -> function
        | Event.Alloc b when where b ->
            let id = b.Block.alloc.Alloc_script.Alloc.id
            and bytes = b.Block.alloc.Alloc_script.Alloc.bytes in
            let+ resident =
              Byte_size.add resident bytes
              |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
                  `Peak_bytes_overflow id)
            in
            ( resident,
              Tensor_id.Map.add id bytes live,
              Byte_size.max peak resident )
        | Event.Free id -> (
            match Tensor_id.Map.find_opt id live with
            | None -> Err.return (resident, live, peak)
            | Some bytes ->
                (* [resident] holds [bytes] since its alloc. *)
                Err.return
                  ( Err.or_raise ~pp_error (Byte_size.sub resident bytes),
                    Tensor_id.Map.remove id live,
                    peak ))
        | Event.Alloc _ | Event.Boundary _ | Event.Node _ ->
            Err.return (resident, live, peak))
      (Byte_size.zero, Tensor_id.Map.empty, Byte_size.zero)
      t.events
  in
  peak

let pp ppf t =
  Format.fprintf ppf "@[<v>%a; policy %a@,%a@]" Config.pp t.config
    Alignment_policy.pp t.policy
    (Format.pp_print_list ~pp_sep:Format.pp_print_cut Event.pp)
    t.events
