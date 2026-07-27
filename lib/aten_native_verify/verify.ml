(* Element-wise comparison between ATen and native tensor outputs.

   Verification order:
     1. shapes must match (right-aligned ATen rank vs 6D native frame)
     2. element types must match (ATen scalar_type vs native payload.fmt)
     3. payload scanned in 6D order; mismatches reported with coordinates

   Float payloads (F32) allow a tolerance; integer payloads (I32, I64) require
   an exact match.  Other ATen/native format combinations are reported as
   Fmt_mismatch. *)

open Schema_runtime
open Core.Syntax

let max_points = 10

module O = Aten_c.Aten_operations

let default_atol = 1e-5

(* Most bridge-covered ops are either pointwise or otherwise deterministic
   enough to compare at a tight tolerance. Matrix/reduction-heavy ops can
   differ slightly in accumulation order between ATen and the native evaluator,
   so widen them selectively rather than weakening verification globally. *)
let atol_for_target = function
  | "torch.ops.aten.addmm.default" -> 1e-4
  | _ -> default_atol

type point = { coord : Vec6.coord; aten_val : float; native_val : float }

type error =
  | Output_count of { expected : int; got : int }
  | Missing_output of { name : string }
  | Shape_mismatch of { output : string; aten : int array; native : Vec6.shape }
  | Fmt_mismatch of {
      output : string;
      aten_dtype : Aten_scalar_type.t;
      native_fmt : string;
    }
  | Payload_mismatch of { output : string; total : int; points : point list }

let scalar_type_name = function
  | Aten_scalar_type.Byte -> "uint8"
  | Aten_scalar_type.Char -> "int8"
  | Aten_scalar_type.Short -> "int16"
  | Aten_scalar_type.Int -> "int32"
  | Aten_scalar_type.Long -> "int64"
  | Aten_scalar_type.Half -> "float16"
  | Aten_scalar_type.Float -> "float32"
  | Aten_scalar_type.Double -> "float64"
  | Aten_scalar_type.Bool -> "bool"

let pp_point ppf p =
  Fmt.pf ppf "%a aten=%g native=%g" Vec6.pp_coord p.coord p.aten_val
    p.native_val

let pp_dims = Fmt.list ~sep:(Fmt.any "x") Fmt.int

let pp_error ppf = function
  | Output_count { expected; got } ->
      Fmt.pf ppf "output count: expected %d got %d" expected got
  | Missing_output { name } -> Fmt.pf ppf "missing ATen output %S" name
  | Shape_mismatch { output; aten; native } ->
      Fmt.pf ppf "%s: shape mismatch: aten [%a] native %a" output pp_dims
        (Array.to_list aten) Vec6.pp_shape native
  | Fmt_mismatch { output; aten_dtype; native_fmt } ->
      Fmt.pf ppf "%s: type mismatch: aten %s native %s" output
        (scalar_type_name aten_dtype)
        native_fmt
  | Payload_mismatch { output; total; points } ->
      Fmt.pf ppf "%s: payload mismatch (%d/%d shown):@ @[<v>%a@]" output
        (List.length points) total
        (Fmt.list ~sep:Fmt.cut pp_point)
        points

(* Scan the 6D shape calling [f coord i] at every element; collect up to
   [max_points] points where [f] returns [Some], counting the total. *)
let scan shape f =
  let points = ref [] and total = ref 0 in
  Vec6.iter shape (fun coord ->
      let i = (Vec6.offset shape coord :> int) in
      match f coord i with
      | None -> ()
      | Some pt ->
          incr total;
          if !total <= max_points then points := pt :: !points);
  (List.rev !points, !total)

(* Do two f32 elements disagree?

   The tolerance test alone silently passes every NaN: [nan -. x] is [nan] and
   [nan > atol] is false, so a native NaN where ATen produced a finite value —
   or the reverse — read as agreement, and any test relying on this comparator
   to check NaN behaviour proved nothing. NaN-ness is therefore compared first,
   as an exact property, and only two finite values reach the tolerance.

   Both-NaN counts as agreement: NaN has no payload to compare here (the engine
   does not distinguish quiet NaN bit patterns), and ATen itself produces NaN
   deliberately in cases the native side must reproduce — a clamp with a NaN
   bound fills its whole result with NaN. Infinities compare by equality for the
   same reason: [inf -. inf] is [nan], so the tolerance test would pass two
   infinities of OPPOSITE sign. *)
let float_differs ~atol av nv =
  match (Float.is_nan av, Float.is_nan nv) with
  | true, true -> false
  | true, false | false, true -> true
  | false, false ->
      if Float.is_finite av && Float.is_finite nv then
        Float.abs (av -. nv) > atol
      else not (Float.equal av nv)

(* Some ATen ops (notably permute/view/expand/select) return non-contiguous
   views. [Aten_tensor.data] exposes a flat buffer and therefore only reflects
   logical element order for contiguous tensors, so verification materializes
   such views before scanning them element-wise. *)
let logical_tensor t =
  if Aten_tensor.is_contiguous t then t
  else Aten_tensor.manage (O.contiguous t Aten_memory_format.Contiguous)

let compare_tensors ~atol ~output aten_t native_t =
  let (Tensor.Tensor native_r) = native_t in
  let aten_t = logical_tensor aten_t in
  (* 1. Shape *)
  let aten_arr = Aten_tensor.shape aten_t in
  let rank = Array.length aten_arr in
  let native_arr = Aten_shape.to_aten ~rank native_r.shape in
  let* () =
    if not (aten_arr = native_arr) then
      Core.fail
        (Shape_mismatch { output; aten = aten_arr; native = native_r.shape })
    else Core.return ()
  in
  (* 2. Format + payload *)
  let aten_dtype = Aten_tensor.scalar_type aten_t in
  match (aten_dtype, native_r.payload.fmt) with
  | Aten_scalar_type.Float, Payload.F32 ->
      let aten_ba = Option.get (Aten_tensor.data Aten_dtype.float32 aten_t) in
      let native_ba = native_r.payload.data in
      let points, total =
        scan native_r.shape (fun coord i ->
            let av = aten_ba.{i} and nv = native_ba.{i} in
            if float_differs ~atol av nv then
              Some { coord; aten_val = av; native_val = nv }
            else None)
      in
      if total = 0 then Core.return ()
      else Core.fail (Payload_mismatch { output; total; points })
  | Aten_scalar_type.Long, Payload.I64 ->
      let aten_ba = Option.get (Aten_tensor.data Aten_dtype.int64 aten_t) in
      let native_ba = native_r.payload.data in
      let points, total =
        scan native_r.shape (fun coord i ->
            let av = aten_ba.{i} and nv = native_ba.{i} in
            if not (Int64.equal av nv) then
              Some
                {
                  coord;
                  aten_val = Int64.to_float av;
                  native_val = Int64.to_float nv;
                }
            else None)
      in
      if total = 0 then Core.return ()
      else Core.fail (Payload_mismatch { output; total; points })
  | Aten_scalar_type.Int, Payload.I32 ->
      let aten_ba = Option.get (Aten_tensor.data Aten_dtype.int32 aten_t) in
      let native_ba = native_r.payload.data in
      let points, total =
        scan native_r.shape (fun coord i ->
            let av = aten_ba.{i} and nv = native_ba.{i} in
            if not (Int32.equal av nv) then
              Some
                {
                  coord;
                  aten_val = Int32.to_float av;
                  native_val = Int32.to_float nv;
                }
            else None)
      in
      if total = 0 then Core.return ()
      else Core.fail (Payload_mismatch { output; total; points })
  | _ ->
      Core.fail
        (Fmt_mismatch
           {
             output;
             aten_dtype;
             native_fmt = Payload.fmt_name native_r.payload.fmt;
           })

(* Verify the tensor outputs [native_outputs] the bridge exposes against the
   LEADING tensor outputs of [node].  The native bridge may expose FEWER outputs
   than the ATen op: a dead output (e.g. max_pool2d_with_indices' int64 argmax
   indices, which the native F32 engine can't compare anyway) is dropped /
   routed to a Discard sink, so it is simply not verified.  Only exposing MORE
   outputs than the op has is an error.  Returns one Core.Error.t per output
   that fails; empty list means all compared outputs matched. *)
let verify_node ~atol ~aten_env (node : Pytorch_types.Node.t) native_outputs =
  let open Pytorch_types in
  let out_names =
    List.filter_map
      (function Argument.Tensor ta -> Some ta.TensorArgument.name | _ -> None)
      node.outputs
  in
  if List.length native_outputs > List.length out_names then
    [
      Core.Error.make
        (Output_count
           {
             expected = List.length out_names;
             got = List.length native_outputs;
           });
    ]
  else
    let k = List.length native_outputs in
    let leading = List.filteri (fun i _ -> i < k) out_names in
    List.filter_map
      (fun (name, native_t) ->
        match String_map.find_opt name aten_env with
        | None -> Some (Core.Error.make (Missing_output { name }))
        | Some aten_t -> (
            match compare_tensors ~atol ~output:name aten_t native_t with
            | Ok () -> None
            | Error e -> Some e))
      (List.combine leading native_outputs)

let report ppf op errors =
  List.iter
    (fun e -> Fmt.pf ppf "[verify] %s: %a@." op pp_error e.Core.Error.kind)
    errors
