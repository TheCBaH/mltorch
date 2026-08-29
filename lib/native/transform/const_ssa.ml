open Graph_ir

module Value_id = struct
  type t = Tensor_id.t

  let of_tensor_id id = id
  let to_tensor_id id = id
  let compare = Tensor_id.compare
  let equal = Tensor_id.equal
  let pp = Tensor_id.pp

  module Map = Map.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

module Capture = struct
  type t = string

  let of_string s = s
  let to_string s = s
  let compare = String.compare
  let equal = String.equal
  let pp fmt s = Format.fprintf fmt "%S" s
end

type leaf =
  | Captured of Capture.t
  | Literal of Tensor.packed
  | Opaque_materialized of Tensor.packed

type definition =
  | Apply of { op : Graph_ir.op; output : Tensor_sig.t }
  | Leaf of { leaf : leaf; output : Tensor_sig.t }

type error =
  [ `Duplicate_definition of Value_id.t
  | `Invalid_literal of Value_id.t
  | `Missing_operand of Value_id.t * Value_id.t
  | `Output_id_mismatch of Value_id.t * Tensor_id.t
  | `Output_signature_mismatch of Value_id.t
  | `Shape of Graph_shape.error
  | `Unsupported_op of string ]

type t = definition Value_id.Map.t

let empty = Value_id.Map.empty
let find t id = Value_id.Map.find_opt id t
let bindings = Value_id.Map.bindings
let output = function Apply { output; _ } | Leaf { output; _ } -> output
let signature_of t id = Option.map output (find t id)
let sig_of = signature_of

let operands t id =
  match find t id with
  | Some (Apply { op; _ }) ->
      List.map Value_id.of_tensor_id (Graph_ir.operands op)
  | Some (Leaf _) | None -> []

let fmt_name (Payload.Fmt fmt) = Payload.fmt_name fmt

let same_shape a b =
  List.for_all
    (fun axis -> Dim.equal (Vec6.get a axis) (Vec6.get b axis))
    Axis.all

let payload_matches (sg : Tensor_sig.t) (Tensor.Tensor payload) =
  same_shape sg.shape payload.Tensor.shape
  && String.equal (fmt_name sg.fmt)
       (Payload.fmt_name payload.Tensor.payload.fmt)
  &&
  match (sg.quant, payload.Tensor.payload.quant) with
  | None, Payload.No_quant -> true
  | Some q, Payload.Quant q' -> q = q'
  | _ -> false

let allows = function
  | Graph_ir.Add _ | Graph_ir.Sub _ | Graph_ir.Mul _ | Graph_ir.Div _
  | Graph_ir.Sqrt _ | Graph_ir.Permute _ ->
      true
  | _ -> false

let pp_error ppf : [< error ] -> unit = function
  | `Duplicate_definition id ->
      Fmt.pf ppf "duplicate Const-SSA definition for %a" Value_id.pp id
  | `Invalid_literal id ->
      Fmt.pf ppf "literal for %a does not match its declared signature"
        Value_id.pp id
  | `Missing_operand (value, operand) ->
      Fmt.pf ppf "Const-SSA value %a refers to undefined operand %a" Value_id.pp
        value Value_id.pp operand
  | `Output_id_mismatch (value, output) ->
      Fmt.pf ppf "Const-SSA value %a declares output %a" Value_id.pp value
        Tensor_id.pp output
  | `Output_signature_mismatch id ->
      Fmt.pf ppf
        "Const-SSA output signature for %a disagrees with its operation"
        Value_id.pp id
  | `Shape e -> Graph_shape.pp_error ppf e
  | `Unsupported_op name -> Fmt.pf ppf "%s is not a Const-SSA operation" name

let validate_definition t id definition =
  let output = output definition in
  if not (Tensor_id.equal (Value_id.to_tensor_id id) output.Tensor_sig.id) then
    Err.fail (`Output_id_mismatch (id, output.id))
  else
    match definition with
    | Leaf { leaf = Literal payload | Opaque_materialized payload; output } ->
        if payload_matches output payload then Err.return ()
        else Err.fail (`Invalid_literal id)
    | Leaf { leaf = Captured _; _ } -> Err.return ()
    | Apply { op; output } -> (
        if not (allows op) then Err.fail (`Unsupported_op (Graph_ir.op_name op))
        else
          let open Err.Syntax in
          let* () =
            Err.List.fold_left
              (fun () operand ->
                let operand = Value_id.of_tensor_id operand in
                if Option.is_some (signature_of t operand) then Err.return ()
                else Err.fail (`Missing_operand (id, operand)))
              () (Graph_ir.operands op)
          in
          let* inferred =
            Graph_shape.output_shape op ~sig_of:(fun operand ->
                match signature_of t (Value_id.of_tensor_id operand) with
                | Some sg -> Err.return sg
                | None -> Err.fail (`Missing_tensor_sig operand))
            |> Err.map_error (fun e -> (`Shape e : error))
          in
          match inferred with
          | [ shape ] when same_shape shape output.shape -> Err.return ()
          | [ _ ] -> Err.fail (`Output_signature_mismatch id)
          | _ -> Err.fail (`Output_signature_mismatch id))

let validate t =
  let open Err.Syntax in
  Value_id.Map.fold
    (fun id definition checked ->
      let* () = checked in
      validate_definition t id definition)
    t (Err.return ())

let add t ~id definition =
  if Value_id.Map.mem id t then Err.fail (`Duplicate_definition id)
  else
    let t = Value_id.Map.add id definition t in
    let open Err.Syntax in
    let* () = validate_definition t id definition in
    Err.return t

let pp_leaf fmt = function
  | Captured capture -> Fmt.pf fmt "captured %a" Capture.pp capture
  | Literal _ -> Fmt.string fmt "literal"
  | Opaque_materialized _ -> Fmt.string fmt "opaque-materialized"

let pp fmt t =
  let pp_definition fmt (id, definition) =
    match definition with
    | Leaf { leaf; _ } -> Fmt.pf fmt "%a = %a" Value_id.pp id pp_leaf leaf
    | Apply { op; _ } ->
        Fmt.pf fmt "%a = %a" Value_id.pp id
          (Graph_ir.pp_op_with ~pp_ref:Value_id.pp)
          op
  in
  Fmt.pf fmt "@[<v>%a@]" Fmt.(list ~sep:cut pp_definition) (bindings t)
