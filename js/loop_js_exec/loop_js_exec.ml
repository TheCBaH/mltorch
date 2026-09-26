open Js_of_ocaml
open Loop_ir
module Failure = Loop_js_failure

type error =
  [ Loop_interp.error | `Js_compile of string | `Js_exception of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Js_compile m -> Fmt.pf ppf "the engine refused the emitted source: %s" m
  | `Js_exception m -> Fmt.pf ppf "a host exception escaped the kernel: %s" m
  | #Loop_interp.error as e -> Loop_interp.pp_error ppf e

let global = Js.Unsafe.global
let get = Js.Unsafe.get
let inject = Js.Unsafe.inject

(* jsoo backs a Bigarray with a typed array, held in [.data]; the external
   returns it for any kind. *)
external ba_data :
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> Js.Unsafe.any
  = "caml_ba_to_typed_array"

let data_of (a : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t) =
  ba_data (Bigarray.genarray_of_array1 a)

let little_endian =
  Js.to_bool
    (Js.Unsafe.coerce
       (Js.Unsafe.js_expr "new Uint8Array(new Uint16Array([1]).buffer)[0] === 1"))

(* jsoo stores an [int64] Bigarray as an [Int32Array] of [lo, hi] pairs; on a
   little-endian host a [BigInt64Array] over the same bytes reads and writes it
   exactly. A runtime detail of jsoo, pinned by the alias round-trip test. *)
let big_view data =
  let length = Js.float_of_number (get data "length") in
  Js.Unsafe.new_obj
    (get global "BigInt64Array")
    [|
      inject (get data "buffer");
      inject (get data "byteOffset");
      inject (length /. 2.);
    |]

let int64_view a = big_view (data_of a)

type compiled = {
  program : Loop_program.t;
  sites : Loop_failure.t array;
  kernel : Js.Unsafe.any;
}

let memo : (string, Js.Unsafe.any) Hashtbl.t = Hashtbl.create 16

(* [new Function(body)()]: the factory body defines the kernel and returns it. A
   [SyntaxError] is the engine's verdict, not ours. *)
let compile_source source =
  match Hashtbl.find_opt memo source with
  | Some k -> Err.return k
  | None -> (
      match
        Js.Unsafe.fun_call
          (Js.Unsafe.new_obj (get global "Function")
             [| inject (Js.string source) |])
          [||]
      with
      | k ->
          Hashtbl.add memo source k;
          Err.return k
      | exception Js.Js_error.Exn e ->
          Err.import (fun m -> `Js_compile m) (Error (Js.Js_error.to_string e)))

let compile_as (program : Loop_program.t) source =
  match compile_source source with
  | Ok kernel ->
      Err.return { program; sites = Loop_js_failure.sites program; kernel }
  | Error e -> Error e

let compile program =
  compile_as program (Js_print.factory_body (Loop_js.to_ast program))

(* ---- decoding a failure record --------------------------------------------- *)

let field o f = get o (Failure.Field.to_string f)
let number o f = Js.float_of_number (Js.Unsafe.coerce (field o f))
let string o f = Js.to_string (Js.Unsafe.coerce (field o f))
let boolean o f = Js.to_bool (Js.Unsafe.coerce (field o f))

(* A record's integer field: bounded before it is narrowed, since a js_of_ocaml
   [int] is 32 bits. *)
let int_field o f =
  let x = number o f in
  if Float.is_integer x && Float.abs x <= 2147483647. then Ok (int_of_float x)
  else
    Error
      (Printf.sprintf "field %s is not a 32-bit integer"
         (Failure.Field.to_string f))

let ( let* ) = Result.bind

let site_failure compiled o =
  let* site = int_field o Failure.Field.Site in
  if site >= 0 && site < Array.length compiled.sites then
    Ok compiled.sites.(site)
  else Error "failure site out of range"

let decode compiled o : (error, string) result =
  let kind = Js.to_string (Js.Unsafe.coerce (get o Failure.kind_key)) in
  match Failure.Kind.of_string kind with
  | None -> Error ("unknown failure kind " ^ kind)
  | Some k -> (
      match k with
      | Failure.Kind.Coord_out_of_range ->
          let* buffer = int_field o Failure.Field.Buffer in
          let* axis = int_field o Failure.Field.Axis in
          let* index = int_field o Failure.Field.Index in
          let coord = field o Failure.Field.Coord in
          let component i =
            let x =
              Js.float_of_number
                (Js.Unsafe.coerce (get coord (string_of_int i)))
            in
            if Float.is_integer x && Float.abs x <= 2147483647. then
              Ok (int_of_float x)
            else Error "coord component is not a 32-bit integer"
          in
          let* n = component 0 in
          let* t = component 1 in
          let* d = component 2 in
          let* h = component 3 in
          let* w = component 4 in
          let* c = component 5 in
          if axis < 0 || axis >= List.length Expr.Axis.all then
            Error "axis out of range"
          else
            Ok
              (`Coord_out_of_range
                 ( Expr_bridge.source_of_id (Tensor_id.of_int buffer),
                   List.nth Expr.Axis.all axis,
                   index,
                   Expr.Coord.make ~n ~t ~d ~h ~w ~c ))
      | Failure.Kind.Defect -> Error "coord_failure found no axis out of range"
      | Failure.Kind.Gather_index_out_of_range -> (
          let* extent = int_field o Failure.Field.Extent in
          match Int64.of_string_opt (string o Failure.Field.Raw) with
          | Some raw ->
              Ok
                (`Gather_index_out_of_range
                   { Expr.Eval.Gather_index_out_of_range.raw; extent })
          | None -> Error "gather raw is not an int64")
      | Failure.Kind.I64_division_by_zero -> Ok `I64_division_by_zero
      | Failure.Kind.I64_division_overflow -> Ok `I64_division_overflow
      | Failure.Kind.I64_from_float_infinite -> Ok `I64_from_float_infinite
      | Failure.Kind.I64_from_float_nan -> Ok `I64_from_float_nan
      | Failure.Kind.I64_from_float_out_of_range ->
          Ok (`I64_from_float_out_of_range (number o Failure.Field.Value))
      | Failure.Kind.Index_overflow -> (
          let* lhs = int_field o Failure.Field.Lhs in
          let* rhs = int_field o Failure.Field.Rhs in
          match Failure.Overflow_op.of_string (string o Failure.Field.Op) with
          | Some Failure.Overflow_op.Add ->
              Ok (`Index_overflow { Expr.Index_overflow.op = `Add; lhs; rhs })
          | Some Failure.Overflow_op.Mul ->
              Ok (`Index_overflow { Expr.Index_overflow.op = `Mul; lhs; rhs })
          | None -> Error "unknown index overflow operator")
      | Failure.Kind.Scan_meter -> (
          match Failure.Meter.of_string (string o Failure.Field.Which) with
          | Some Failure.Meter.Updates_exhausted ->
              Ok
                (`Scan_meter
                   (Expr.Scan_meter.Updates_exhausted
                      { limit = Int64.of_float (number o Failure.Field.Limit) }))
          | Some Failure.Meter.State_over_limit ->
              let* limit = int_field o Failure.Field.Limit in
              Ok (`Scan_meter (Expr.Scan_meter.State_over_limit { limit }))
          | None -> Error "unknown scan meter kind")
      | Failure.Kind.Scan_projection -> (
          let* site = site_failure compiled o in
          let* row = int_field o Failure.Field.Row in
          let* lane = int_field o Failure.Field.Lane in
          let* extent = int_field o Failure.Field.Extent in
          let bounds local =
            {
              Expr.Eval.Scan_bounds.projection =
                { Expr.Eval.Scan_projection.local; row; lane };
              extent;
            }
          in
          ignore (boolean o Failure.Field.Cached);
          match
            (Failure.Projection.of_string (string o Failure.Field.Which), site)
          with
          | ( Some Failure.Projection.Lane,
              Loop_failure.Scan_lane_out_of_range { local; _ } ) ->
              Ok (`Scan_projection (Expr.Eval.Lane_out_of_range (bounds local)))
          | ( Some Failure.Projection.Row,
              Loop_failure.Scan_row_out_of_range { local; _ } ) ->
              Ok (`Scan_projection (Expr.Eval.Row_out_of_range (bounds local)))
          | _ -> Error "scan projection record disagrees with its site")
      | Failure.Kind.Unbound_local -> (
          let* site = site_failure compiled o in
          match site with
          | Loop_failure.Local_out_of_range { local; _ } ->
              Ok (`Unbound_local local)
          | _ -> Error "unbound_local record disagrees with its site"))

(* ---- running ---------------------------------------------------------------- *)

let bind_buffers (p : Loop_program.t) ~bind =
  List.fold_left
    (fun acc (b : Loop_buffer.t) ->
      let* acc = acc in
      match b.Loop_buffer.role with
      | Loop_buffer.Input -> (
          match bind b.Loop_buffer.id with
          | None -> Error (`Unbound_input b.Loop_buffer.id)
          | Some tensor -> (
              match
                Err.payload
                  (Kernel_eval.check_binding b.Loop_buffer.id b.Loop_buffer.sg
                     tensor)
              with
              | Error (`Binding_mismatch m) -> Error (`Binding_mismatch m)
              | Ok () -> Ok (Tensor_id.Map.add b.Loop_buffer.id tensor acc)))
      | Loop_buffer.Output | Loop_buffer.Scratch ->
          Ok (Tensor_id.Map.add b.Loop_buffer.id (Loop_interp.allocate b) acc))
    (Ok Tensor_id.Map.empty) p.Loop_program.buffers

(* The typed array a buffer's storage is passed as: the Bigarray's own data, or
   for int64 the [BigInt64Array] alias. The emitter names the constructor it
   expects; a disagreement is a [`Binding_mismatch], never a wrong read. *)
let argument (b : Loop_buffer.t) (Tensor.Tensor t) =
  let data = data_of t.Tensor.payload.Payload.data in
  let arg =
    match Loop_js.typed_array b with
    | "BigInt64Array" -> big_view data
    | _ -> data
  in
  if Js.instanceof arg (get global (Loop_js.typed_array b)) then Ok arg
  else
    Error
      (`Binding_mismatch
         {
           Kernel_eval.Binding_mismatch.id = b.Loop_buffer.id;
           kind = Kernel_eval.Binding_mismatch.Format;
         })

let run compiled ~bind =
  let p = compiled.program in
  let result =
    if not little_endian then
      Error (`Js_exception "a big-endian host is not supported")
    else
      let* tensors = bind_buffers p ~bind in
      let* args =
        List.fold_left
          (fun acc (b : Loop_buffer.t) ->
            let* acc = acc in
            let* a = argument b (Tensor_id.Map.find b.Loop_buffer.id tensors) in
            Ok (a :: acc))
          (Ok []) p.Loop_program.buffers
      in
      match
        Js.Unsafe.fun_call compiled.kernel (Array.of_list (List.rev args))
      with
      | exception Js.Js_error.Exn e ->
          Error (`Js_exception (Js.Js_error.to_string e))
      | r when not (Js.Unsafe.equals r (inject Js.null)) -> (
          match decode compiled r with
          | Ok row -> Error row
          | Error m -> Error (`Js_exception m))
      | _ ->
          Ok
            (List.fold_left
               (fun acc (b : Loop_buffer.t) ->
                 match b.Loop_buffer.role with
                 | Loop_buffer.Output ->
                     Tensor_id.Map.add b.Loop_buffer.id
                       (Tensor_id.Map.find b.Loop_buffer.id tensors)
                       acc
                 | Loop_buffer.Input | Loop_buffer.Scratch -> acc)
               Tensor_id.Map.empty p.Loop_program.buffers)
  in
  match result with Ok m -> Err.return m | Error e -> Err.fail (e :> error)

let exec program ~bind =
  match Err.payload (compile program) with
  | Error e -> Err.fail (e :> error)
  | Ok compiled -> run compiled ~bind
