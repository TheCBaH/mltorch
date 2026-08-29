(* Element-wise comparison between ATen and native tensor outputs.

   Verification order:
     1. shapes must match (right-aligned ATen rank vs 6D native frame)
     2. element types must match (ATen scalar_type vs native payload.fmt)
     3. payload scanned in 6D order; mismatches reported with coordinates

   Float payloads (F32) allow a tolerance; integer payloads (I32, I64) require
   an exact match.  Other ATen/native format combinations are reported as
   Fmt_mismatch. *)

open Schema_runtime
open Err.Syntax

let max_points = 10
let default_atol = 1e-5

(* Most bridge-covered ops are either pointwise or otherwise deterministic
   enough to compare at a tight tolerance. Matrix/reduction-heavy ops can
   differ slightly in accumulation order between ATen and the native evaluator,
   so widen them selectively rather than weakening verification globally. *)
let atol_for_target = function
  | "torch.ops.aten.addmm.default" -> 1e-4
  (* op8-impl.md commit 4 (F4): the CPU flash kernel is a blocked, tiled,
     online-softmax reassociation of the same math this row implements
     (split-sqrt scale, unconditional safe-softmax) -- a real numerical
     divergence, not a bug, so it gets its own measured entry rather than
     inheriting [default_atol] by coincidence. Measured (scratch probe,
     op8-impl.md commit 0 step 2 and commit 4 step 3, never committed): the
     worst observed |flash - math| across Recipe_sdpa's full shape space
     (3240 samples) was 4.1e-7; the worst across large-magnitude logits
     (op8-impl.md's max-subtraction-stability case, x20 scale) was 1.5e-8.
     Both stay comfortably under [default_atol]; this entry is pinned
     independently of it so a future unrelated change to [default_atol]
     cannot silently loosen or tighten what this target was actually
     verified against. This is defensible as ONE entry only because the
     recipe's types make an off-oracle (non-flash) configuration
     unrepresentable (F4/F9) -- if a future recipe widening reintroduces the
     math backend, it needs its own policy, not a wider single number. *)
  | "torch.ops.aten.scaled_dot_product_attention.default" -> 1e-5
  | _ -> default_atol

type point = { coord : Vec6.coord; aten_val : float; native_val : float }

type error =
  | Fmt_mismatch of {
      output : string;
      aten_dtype : Aten_scalar_type.t;
      native_fmt : string;
    }
  | Missing_output of { name : string }
  | Output_count of { expected : int; got : int }
  | Payload_mismatch of { output : string; total : int; points : point list }
  | Shape_mismatch of { output : string; aten : int array; native : Vec6.shape }

let scalar_type_name = function
  | Aten_scalar_type.Bool -> "bool"
  | Aten_scalar_type.Byte -> "uint8"
  | Aten_scalar_type.Char -> "int8"
  | Aten_scalar_type.Double -> "float64"
  | Aten_scalar_type.Float -> "float32"
  | Aten_scalar_type.Half -> "float16"
  | Aten_scalar_type.Int -> "int32"
  | Aten_scalar_type.Long -> "int64"
  | Aten_scalar_type.Short -> "int16"

let pp_point ppf p =
  Fmt.pf ppf "%a aten=%g native=%g" Vec6.pp_coord p.coord p.aten_val
    p.native_val

let pp_dims = Fmt.list ~sep:(Fmt.any "x") Fmt.int

let pp_error ppf = function
  | Fmt_mismatch { output; aten_dtype; native_fmt } ->
      Fmt.pf ppf "%s: type mismatch: aten %s native %s" output
        (scalar_type_name aten_dtype)
        native_fmt
  | Missing_output { name } -> Fmt.pf ppf "missing ATen output %S" name
  | Output_count { expected; got } ->
      Fmt.pf ppf "output count: expected %d got %d" expected got
  | Payload_mismatch { output; total; points } ->
      Fmt.pf ppf "%s: payload mismatch (%d/%d shown):@ @[<v>%a@]" output
        (List.length points) total
        (Fmt.list ~sep:Fmt.cut pp_point)
        points
  | Shape_mismatch { output; aten; native } ->
      Fmt.pf ppf "%s: shape mismatch: aten [%a] native %a" output pp_dims
        (Array.to_list aten) Vec6.pp_shape native

(* Scan the 6D shape calling [f coord i] at every element; collect up to
   [max_points] points where [f] returns [Some], counting the total. *)
let scan shape f =
  let points = ref [] and total = ref 0 in
  Vec6.iter shape (fun coord ->
      let i = (Vec6.offset shape coord :> int) in
      Option.iter
        (fun pt ->
          incr total;
          if !total <= max_points then points := pt :: !points)
        (f coord i));
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

(* Some ATen ops (notably permute/view/expand/select/unbind) return views.
   [Aten_tensor.data] exposes the flat STORAGE and so reflects logical element
   order only for a contiguous tensor at offset zero, so verification
   materializes anything else before scanning it element-wise. Guarding on
   [is_contiguous] alone silently compared a select view against the wrong
   region of its storage. *)
let logical_tensor = Aten_tensor.materialize_for_raw_read

let compare_tensors ~atol ~output aten_t native_t =
  let (Tensor.Tensor native_r) = native_t in
  let aten_t = logical_tensor aten_t in
  (* 1. Shape *)
  let aten_arr = Aten_tensor.shape aten_t in
  let rank = Array.length aten_arr in
  let native_arr = Aten_shape.to_aten ~rank native_r.shape in
  let* () =
    if not (aten_arr = native_arr) then
      Err.fail
        (Shape_mismatch { output; aten = aten_arr; native = native_r.shape })
    else Err.return ()
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
      if total = 0 then Err.return ()
      else Err.fail (Payload_mismatch { output; total; points })
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
      if total = 0 then Err.return ()
      else Err.fail (Payload_mismatch { output; total; points })
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
      if total = 0 then Err.return ()
      else Err.fail (Payload_mismatch { output; total; points })
  | _ ->
      Err.fail
        (Fmt_mismatch
           {
             output;
             aten_dtype;
             native_fmt = Payload.fmt_name native_r.payload.fmt;
           })

(* A FIXED tuple's arity is the op's, so the bridge may legitimately expose
   fewer outputs than it: a dead output (max_pool2d_with_indices' int64 argmax
   indices, which the native F32 engine cannot compare anyway) is dropped or
   routed to a Discard sink, and is simply not verified.

   A DYNAMIC `Tensor[]` return is different. Its arity is not the op's, it is
   the input's, so there is no dead-output story and no reason to expose fewer:
   a bridge that returned only the first slice would be WRONG, and under the
   leading-outputs rule it would be reported as matched. That is a false green
   the ordinal-perturbation test cannot see, so exactness is required here.

   Decided by the node's OUTPUT STRUCTURE, never by its target string: a single
   [Argument.Tensors] output is the dynamic shape, and the structure is right
   there to be read. *)
let requires_exact_outputs (node : Pytorch_types.Node.t) =
  match node.Pytorch_types.Node.outputs with
  | [ Pytorch_types.Argument.Tensors _ ] -> true
  | _ -> false

(* Verify the tensor outputs [native_outputs] the bridge exposes against the
   tensor outputs of [node] — all of them for a dynamic list, the LEADING ones
   for a fixed tuple (see above).  Only exposing MORE outputs than the op has is
   an error either way.  Returns one Err.Error.t per output that fails; empty
   list means all compared outputs matched.

   [output_names] flattens a Tensor[] output's names in place, so a
   list-returning op exposes all N here rather than none. *)
let verify_node ~atol ~aten_env (node : Pytorch_types.Node.t) native_outputs =
  let out_names = Interp_decode.output_names node in
  let wrong_count =
    if requires_exact_outputs node then
      List.compare_lengths native_outputs out_names <> 0
    else List.length native_outputs > List.length out_names
  in
  if wrong_count then
    [
      Err.Error.make
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
        | None -> Some (Err.Error.make (Missing_output { name }))
        | Some aten_t -> (
            match compare_tensors ~atol ~output:name aten_t native_t with
            | Ok () -> None
            | Error e -> Some e))
      (List.combine leading native_outputs)

let report ppf op errors =
  List.iter
    (fun e -> Fmt.pf ppf "[verify] %s: %a@." op pp_error (Err.Error.kind e))
    errors
