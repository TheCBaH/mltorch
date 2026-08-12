(* Schema-driven graph-dispatch generator: the dual of Aten_c_type.map_type.

   For a parsed ATen op it emits the interpreter match arm that decodes an
   exported graph node's named arguments (via the Interp_decode helpers) and
   calls the bound op [O.<name>], or returns [None] when any argument/return type
   is outside the supported set (such an op produces no arm and reaches the
   interpreter's failwith if a graph ever uses it). Where map_type lowers a
   schema arg to its ctypes fragments, [decode_arg] pulls those same fragments
   back out of the node: a list arg expands to a (data, length) pair, a Tensor to
   a resolved handle, a Scalar through the scalar decoder, etc. *)

module A = Aten_func_ast

(* Numeric literals: parenthesise negatives so [Constr -1L] is not read as
   subtraction (e.g. hardtanh's min_val=-1). *)
let lit n = if n < 0 then Printf.sprintf "(%d)" n else string_of_int n
let lit64 n = if n < 0 then Printf.sprintf "(%dL)" n else Printf.sprintf "%dL" n

let litf s =
  if String.length s > 0 && s.[0] = '-' then Printf.sprintf "(%s)" s else s

(* The [~default] fragment for an int-list argument. A list literal default
   (stride=[]) renders verbatim; a scalar default (padding=0 on an int[2])
   broadcasts to the list's declared size ([0; 0]); an absent or unsized scalar
   default yields no fragment (ints_arg then falls back to []). *)
let int_list_default default size =
  let render = function
    | [] -> " ~default:[]"
    | l ->
        Printf.sprintf " ~default:[ %s ]" (String.concat "; " (List.map lit l))
  in
  match (default, size) with
  | Some (A.Default.IntList l), _ -> render l
  | Some (A.Default.Int n), Some k -> render (List.init k (fun _ -> n))
  | _ -> ""

(* The OCaml binding name, matching Aten_gen.names: base, or base_overload. *)
let ocaml_name (op : A.t) =
  match op.name.overload with
  | None | Some "" -> op.name.base
  | Some ov -> Printf.sprintf "%s_%s" op.name.base ov

(* The exported node target string, e.g. torch.ops.aten.add.Tensor. The absent
   overload serializes as the "default" overload. *)
let target (op : A.t) =
  let ov =
    match op.name.overload with None | Some "" -> "default" | Some s -> s
  in
  Printf.sprintf "torch.ops.aten.%s.%s" op.name.base ov

(* The first required Tensor argument. It is resolved with [tensor_arg] and bound
   to a let, then used as the broadcast reference (~like) for any later
   Tensor-typed argument that the exporter may have serialized as a bare scalar
   (see Interp_decode.tensor_or_scalar_arg). *)
let anchor_of (args : A.Argument.t list) =
  List.find_map
    (fun (a : A.Argument.t) ->
      match a.ty with A.Type.Base A.Base.Tensor -> Some a.name | _ -> None)
    args

let is_mean_dim_optional_dims (op : A.t) (a : A.Argument.t) =
  String.equal op.name.base "mean"
  &&
  match op.name.overload with
  | Some "dim" -> String.equal a.name "dim"
  | _ -> false

(* Decode one argument to (preamble lets, call-site expressions). A list arg
   yields a preamble binding plus two expressions (data ptr + length), mirroring
   the (data, len) C parameter pair; everything else is a single expression and
   no preamble. Returns [None] for an unsupported type. *)
let decode_arg ~(op : A.t) ~anchor (a : A.Argument.t) =
  let q = Printf.sprintf "%S" a.name in
  let bind var call exprs =
    Some ([ Printf.sprintf "let* %s = %s in" var call ], exprs)
  in
  match a.ty with
  | A.Type.Base A.Base.Tensor -> (
      if Some a.name = anchor then
        bind a.name (Printf.sprintf "(tensor_arg env node %s)" q) [ a.name ]
      else
        match anchor with
        | Some like ->
            bind a.name
              (Printf.sprintf "(tensor_or_scalar_arg env node %s ~like:%s)" q
                 like)
              [ a.name ]
        | None ->
            bind a.name (Printf.sprintf "(tensor_arg env node %s)" q) [ a.name ]
      )
  | A.Type.Optional (A.Type.Base A.Base.Tensor) ->
      bind a.name (Printf.sprintf "(tensor_arg env node %s)" q) [ a.name ]
  | A.Type.Base A.Base.Int | A.Type.Base A.Base.SymInt ->
      let d =
        match a.default with
        | Some (A.Default.Int n) -> Printf.sprintf " ~default:%s" (lit n)
        | _ -> ""
      in
      bind a.name
        (Printf.sprintf "(int_arg%s node %s)" d q)
        [ Printf.sprintf "(Int64.of_int %s)" a.name ]
  | A.Type.Base A.Base.Float ->
      let d =
        match a.default with
        | Some (A.Default.Float s) -> Printf.sprintf " ~default:%s" (litf s)
        | _ -> ""
      in
      bind a.name (Printf.sprintf "(float_arg%s node %s)" d q) [ a.name ]
  | A.Type.Base A.Base.Bool ->
      let d =
        match a.default with
        | Some (A.Default.Bool b) -> Printf.sprintf " ~default:%b" b
        | _ -> ""
      in
      bind a.name (Printf.sprintf "(bool_arg%s node %s)" d q) [ a.name ]
  | A.Type.Base A.Base.Scalar ->
      let d =
        match a.default with
        | Some (A.Default.Int n) ->
            Printf.sprintf " ~default:(Aten_scalar.Int %s)" (lit64 n)
        | Some (A.Default.Float s) ->
            Printf.sprintf " ~default:(Aten_scalar.Float %s)" (litf s)
        | _ -> ""
      in
      bind a.name (Printf.sprintf "(scalar_arg%s node %s)" d q) [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.Scalar) ->
      bind a.name (Printf.sprintf "(scalar_opt_arg node %s)" q) [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.Float) ->
      bind a.name (Printf.sprintf "(float_opt_ptr node %s)" q) [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.Bool) ->
      bind a.name (Printf.sprintf "(bool_opt_arg node %s)" q) [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.MemoryFormat) ->
      bind a.name
        (Printf.sprintf "(memory_format_opt_arg node %s)" q)
        [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.ScalarType) ->
      bind a.name (Printf.sprintf "(scalar_type_opt_arg node %s)" q) [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.Layout) ->
      bind a.name (Printf.sprintf "(layout_opt_arg node %s)" q) [ a.name ]
  | A.Type.Optional (A.Type.Base A.Base.Device) ->
      bind a.name (Printf.sprintf "(device_opt_arg node %s)" q) [ a.name ]
  | A.Type.Base A.Base.Str -> (
      match a.default with
      | Some (A.Default.Str s) ->
          bind a.name
            (Printf.sprintf "(string_arg ~default:%S node %s)" s q)
            [ a.name ]
      | _ -> None)
  (* int[] / SymInt[] and their optional forms all lower to a (data, len) pair.
     [aten.mean.dim] is special: its optional [dim] means "all dims" when
     absent, and ATen treats an empty list the same way, so the spec path must
     decode the missing case to [[]] rather than raising a missing-argument
     error. *)
  | A.Type.List ((A.Type.Base A.Base.Int | A.Type.Base A.Base.SymInt), size)
  | A.Type.Optional
      (A.Type.List ((A.Type.Base A.Base.Int | A.Type.Base A.Base.SymInt), size))
    ->
      let d =
        let d = int_list_default a.default size in
        if String.equal d "" && is_mean_dim_optional_dims op a then
          " ~default:[]"
        else d
      in
      Some
        ( [ Printf.sprintf "let* %s = ints_arg%s node %s in" a.name d q ],
          [
            Printf.sprintf "(arr %s)" a.name;
            Printf.sprintf "(List.length %s)" a.name;
          ] )
  | A.Type.List (A.Type.Base A.Base.Tensor, _) ->
      Some
        ( [ Printf.sprintf "let* %s = tensors_arg env node %s in" a.name q ],
          [
            Printf.sprintf "(tensor_carray %s)" a.name;
            Printf.sprintf "(List.length %s)" a.name;
          ] )
  | _ -> None

(* The match arm for [op], or [None] if its returns or any argument are
   unsupported. *)
let dispatch_arm (op : A.t) : string option =
  if op.arguments.out <> [] then None
  else
    match Aten_c_type.map_returns op.returns with
    | None -> None
    | Some (Aten_c_type.Tensors_ret nret) when nret < 1 || nret > 3 -> None
    | Some (Aten_c_type.Tensors_ret nret) ->
        let args = op.arguments.positional @ op.arguments.kwarg_only in
        let anchor = anchor_of args in
        let decoded = List.map (fun a -> decode_arg ~op ~anchor a) args in
        if List.exists Option.is_none decoded then None
        else
          let decoded = List.map Option.get decoded in
          let preambles = List.concat_map fst decoded in
          let exprs = List.concat_map snd decoded in
          let call =
            Printf.sprintf "O.%s %s" (ocaml_name op) (String.concat " " exprs)
          in
          let body =
            if nret = 1 then Printf.sprintf "bind1 env node (%s)" call
            else
              let reads =
                List.init nret (fun i ->
                    Printf.sprintf "tget out TG.tensors%d_v%d" nret i)
                |> String.concat "; "
              in
              String.concat "\n      "
                [
                  Printf.sprintf "let out = make TG.tensors%d_struct in" nret;
                  Printf.sprintf "let st = %s (addr out) in" call;
                  Printf.sprintf
                    "if st <> 0 then Err.fail (`Aten_runtime_failure (%S, st)) \
                     else"
                    (ocaml_name op);
                  Printf.sprintf "bind_many env node [ %s ]" reads;
                ]
          in
          let lines = preambles @ [ body ] in
          Some
            (Printf.sprintf "  | %S ->\n      %s" (target op)
               (String.concat "\n      " lines))

let banner =
  {|(* @generated by aten_decode_gen (bin/aten_ops_gen.ml) from native_functions.yaml.
   DO NOT EDIT. Regenerate via the dune rule in lib/interp/dune. *)|}

(* Assemble the whole interp_dispatch.ml: the opens, then one match over the
   exported node target with an arm per supported op. Ops the generator cannot
   decode produce no arm and fall through to the failwith. *)
let file (ops : A.t list) : string =
  let arms = List.filter_map dispatch_arm ops in
  Printf.sprintf
    {|%s

[@@@warning "-33"]

open Ctypes
open Pytorch_types
open Interp_decode
open Err.Syntax
module O = Aten_c.Aten_operations
module TG = Aten_types_generated

let tget = Aten_operation_description.tensors_get

let dispatch (env : env) (node : Node.t) =
  match node.target with
%s
  | other -> Err.fail (`Unhandled_op other)
|}
    banner (String.concat "\n" arms)
