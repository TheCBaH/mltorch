(* conv2d/convolution/pool parameter builders for Native_interp, split out of
   native_interp_decode.ml once that file crossed the tracked 1000-line
   ceiling (scripts/check-file-size.sh). Depends on the generic argument-
   decoding helpers there (int_arg, hw2, pos, tensor_meta, ...), reached
   through the [open] below -- consumers of both (native_interp_lower_compute.ml)
   open this module alongside it. *)

open Pytorch_types
open Native_interp_decode

(* [Conv2d.params.in_channels] is the ACTIVATION's channel count: the weight's
   per-group input extent times the group count. Same rule as
   [Op_bridge.make_conv2d_params] and [Conv.Conv2d_padding.to_conv2d_params],
   restated here only because the importer reads serialized metadata where those
   read a live tensor.

   COMPUTED IN int64 AND BOUNDED BEFORE NARROWING, which is the whole point.
   js_of_ocaml's [int] is 32 bits and [Kernel.Limits.Hard.extent] is
   0x8000_0000, so two individually plausible factors can multiply past the
   representable range and WRAP to a small positive number — a silently wrong
   graph rather than a rejected one. Bounding the factors would not catch it:
   the product is its own quantity. [test/native_interp] runs under node
   ([modes best js]), so this is reachable and not a theoretical concern. *)
let conv_in_channels esc ~tensor ~cin ~groups =
  (* Each FACTOR is bounded before the multiplication, not only the product.
     Widening to [int64] does not by itself make the multiplication safe: these
     are raw decoded ints, and on a 63-bit-[int] backend two factors near
     2^62 overflow [Int64.mul] silently and land back inside the range the
     check below accepts. The bound is unobservable under js_of_ocaml, where a
     factor that large is not representable in a 32-bit [int] at all -- which is
     exactly why it cannot be left to the product check. *)
  let over n = Int64.of_int n >= Kernel.Limits.Hard.extent in
  if over cin || over groups then
    malformed esc
      (`Bad_dimension
         {
           tensor;
           fault =
             `Over_max_extent
               (if over cin then Int64.of_int cin else Int64.of_int groups);
         });
  let product = Int64.mul (Int64.of_int cin) (Int64.of_int groups) in
  if product >= Kernel.Limits.Hard.extent then
    malformed esc (`Bad_dimension { tensor; fault = `Over_max_extent product })
  else if product < 1L then
    malformed esc
      (`Bad_dimension
         { tensor; fault = (if product = 0L then `Zero else `Negative cin) })
  else Dim.extent (Int64.to_int product)

(* The exact `conv1d.default` overload -- [conv2d_params]'s single-spatial-axis
   twin, sharing its metadata/dtype decoding and [conv_in_channels]'s bounded
   product, but with [Conv.Conv1d.params]'s own one-window shape (no [h]
   field: this op names no second spatial axis at all, so unlike
   [Conv2d_padding]'s own single-window "same"/"valid" case there is nothing
   here for a stray value to misrepresent). *)
let conv1d_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv1d_weight)
  in
  let _cout, cin, k = sizes_rank_3 esc ~tensor:weight_name sizes in
  let s = w1 esc `Stride (ints_arg esc ~default:[ 1 ] node "stride") in
  let p = w1 esc `Padding (ints_arg esc ~default:[ 0 ] node "padding") in
  let d = w1 esc `Dilation (ints_arg esc ~default:[ 1 ] node "dilation") in
  let groups = int_arg esc ~default:1 node "groups" in
  {
    Conv.Conv1d.w =
      {
        Conv.Conv2d.kernel = extent esc ~op ~param:`Kernel_size k;
        stride = pos esc ~op ~param:`Stride s;
        pad_before = nonneg esc ~op ~param:`Padding p;
        pad_after = nonneg esc ~op ~param:`Padding p;
        dilation = pos esc ~op ~param:`Dilation d;
      };
    in_channels = conv_in_channels esc ~tensor:weight_name ~cin ~groups;
    groups = pos esc ~op ~param:`Groups groups;
  }

(* The exact `conv2d.default` overload, whose [params] record is NOT
   [Convolution]'s: per-axis windows carrying their own kernel extent, an
   activation channel count, and no transposed/output_padding fields. Sharing
   the metadata and H/W decoding with [conv_params] while keeping the two
   records apart is what lets the exact IR node survive to Native4D, which reads
   [Conv2d] directly. *)
let conv2d_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv2d_weight)
  in
  let _cout, cin, kh, kw = sizes_rank_4 esc ~tensor:weight_name sizes in
  let sh, sw = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let ph, pw =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  let dh, dw =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  (* Serialized integer padding is SYMMETRIC — ATen pads both sides of an axis
     equally — so both fields take the one value. The asymmetric form exists for
     [Conv2d_padding]'s "same", which splits an odd total unevenly. *)
  let axis ~kernel ~stride ~pad ~dilation : Conv.Conv2d.axis_window =
    {
      kernel = extent esc ~op ~param:`Kernel_size kernel;
      stride = pos esc ~op ~param:`Stride stride;
      pad_before = nonneg esc ~op ~param:`Padding pad;
      pad_after = nonneg esc ~op ~param:`Padding pad;
      dilation = pos esc ~op ~param:`Dilation dilation;
    }
  in
  {
    Conv.Conv2d.h = axis ~kernel:kh ~stride:sh ~pad:ph ~dilation:dh;
    w = axis ~kernel:kw ~stride:sw ~pad:pw ~dilation:dw;
    in_channels = conv_in_channels esc ~tensor:weight_name ~cin ~groups;
    groups = pos esc ~op ~param:`Groups groups;
  }

(* The exact `conv3d.default` overload -- [conv2d_params]'s three-spatial-axis
   twin, sharing its metadata/dtype decoding and [conv_in_channels]'s bounded
   product, but with [Conv.Conv3d.params]'s own three-window shape (ATen's own
   D/H/W order, matching native's D/H/W directly -- see [perm_conv3d]). *)
let conv3d_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv3d_weight)
  in
  let _cout, cin, kd, kh, kw = sizes_rank_5 esc ~tensor:weight_name sizes in
  let sd, sh, sw =
    dhw3 esc `Stride (ints_arg esc ~default:[ 1; 1; 1 ] node "stride")
  in
  let pd, ph, pw =
    dhw3 esc `Padding (ints_arg esc ~default:[ 0; 0; 0 ] node "padding")
  in
  let dd, dh, dw =
    dhw3 esc `Dilation (ints_arg esc ~default:[ 1; 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  let axis ~kernel ~stride ~pad ~dilation : Conv.Conv2d.axis_window =
    {
      kernel = extent esc ~op ~param:`Kernel_size kernel;
      stride = pos esc ~op ~param:`Stride stride;
      pad_before = nonneg esc ~op ~param:`Padding pad;
      pad_after = nonneg esc ~op ~param:`Padding pad;
      dilation = pos esc ~op ~param:`Dilation dilation;
    }
  in
  {
    Conv.Conv3d.d = axis ~kernel:kd ~stride:sd ~pad:pd ~dilation:dd;
    h = axis ~kernel:kh ~stride:sh ~pad:ph ~dilation:dh;
    w = axis ~kernel:kw ~stride:sw ~pad:pw ~dilation:dw;
    in_channels = conv_in_channels esc ~tensor:weight_name ~cin ~groups;
    groups = pos esc ~op ~param:`Groups groups;
  }

(* The overload whose padding is a MODE rather than a number. Its [params] carry
   neither a kernel extent nor a channel count: [Conv2d_padding.to_conv2d_params]
   derives both from the weight's shape, and that one definition is already
   shared by shape inference, [Compute] and Native4D. Resolving the mode here
   would make a fourth. *)
let conv2d_padding_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  (* Read for its RANK alone -- the extents are shape inference's business here.
     Checked all the same, so the two conv2d arms accept the same weights: a
     rank-3 weight would otherwise be right-aligned into the six-axis frame and
     relayouted as though its leading axis were the output channel. *)
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv2d_padding_weight)
  in
  let _ = sizes_rank_4 esc ~tensor:weight_name sizes in
  let padding =
    (* NOT [Conv.Conv2d_padding.padding_of_string], which [invalid_arg]s on
       anything else (conv.ml:394). This string is model data, so it gets a
       typed row -- the same rule the guarded config constructors follow. *)
    match
      Conv.Conv2d_padding.of_string
        (string_arg esc ~default:"valid" node "padding")
    with
    | Error s -> malformed esc (`Unsupported_padding_mode s)
    | Ok p -> p
  in
  let stride = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let dilation =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  {
    Conv.Conv2d_padding.stride = pos_hw esc ~op ~param:`Stride stride;
    padding;
    dilation = pos_hw esc ~op ~param:`Dilation dilation;
    groups = pos esc ~op ~param:`Groups (int_arg esc ~default:1 node "groups");
  }

let conv_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Convolution_weight)
  in
  let cout, cin, kh, kw = sizes_rank_4 esc ~tensor:weight_name sizes in
  let _ = cout in
  let op = node.Node.target in
  let stride = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let padding =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  let dilation =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  (* [output_padding] IS an argument of this overload -- the schema is
     convolution(..., bool transposed, SymInt[] output_padding, int groups) --
     and forcing it to zero was wrong in both directions. A transposed
     convolution's output extent includes it, so a nonzero value built a smaller
     op than the model asked for; and a NON-transposed one with a nonzero value
     is invalid, which [Convolution.output_shape] rejects (conv.ml:833) and
     discarding the argument let through. [Op_bridge] has always decoded it, so
     the two importers built different ops from one node. *)
  let output_padding =
    hw2 esc `Output_padding
      (ints_arg esc ~default:[ 0; 0 ] node "output_padding")
  in
  ( {
      Conv.Convolution.stride = pos_hw esc ~op ~param:`Stride stride;
      padding = nonneg_hw esc ~op ~param:`Padding padding;
      dilation = pos_hw esc ~op ~param:`Dilation dilation;
      transposed = bool_arg esc node "transposed";
      output_padding = nonneg_hw esc ~op ~param:`Output_padding output_padding;
      groups = pos esc ~op ~param:`Groups groups;
    },
    cin,
    kh,
    kw )

(* Shared by [pool_params] and [avg_pool_params]: both max- and avg-pool's
   [params] carry the same kernel/stride/pad shape. *)
let pool_window_fields esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kh, kw = hw2 esc `Kernel_size (ints_arg esc node "kernel_size") in
  (* Validated BEFORE the stride is defaulted from it. The default makes the two
     the same value, so a kernel of 0 reached the stride's check first and was
     reported as a bad stride -- a diagnostic naming an argument the model never
     supplied. *)
  let kernel =
    {
      Op_config.Hw.h = extent esc ~op ~param:`Kernel_size kh;
      w = extent esc ~op ~param:`Kernel_size kw;
    }
  in
  (* An EMPTY stride list means "same as the kernel" and is a different
     spelling from the argument being absent, so both normalize here. Same rule
     as [Op_bridge.pool_stride] (op_bridge.ml:371). *)
  let stride =
    match ints_arg esc ~default:[ kh; kw ] node "stride" with
    | [] -> [ kh; kw ]
    | s -> s
  in
  let stride = hw2 esc `Stride stride in
  let padding =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  ( kernel,
    pos_hw esc ~op ~param:`Stride stride,
    nonneg_hw esc ~op ~param:`Padding padding )

(* [max_pool2d.default]/[max_pool2d_with_indices.default]'s own [dilation]:
   REJECTED, not carried -- [Pool.MaxPool2d.params] has no field for it, so a
   non-default value would compute a different op under the right name.
   Extending the native IR is warranted only on a measured need, and no model
   this repository can download serialises either pooling target with a
   non-default dilation, so there is nothing to measure and a rejection is the
   honest answer. [avg_pool2d.default] has no [dilation] argument at all (see
   [avg_pool_params]), so it needs no analogous check. *)
let pool_params esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kernel, stride, pad = pool_window_fields esc node in
  (* NORMALIZED FIRST, then compared against the only value the params can hold.
     Testing "does some element differ from 1" accepted [] and [1;1;1] as well
     as [1;1] -- and ATen refuses both ("dilation must be either a single int,
     or a tuple of two ints"), so the arity check every other H/W argument gets
     was the one thing standing between a refused node and a silent drop. *)
  let dilation = ints_arg esc ~default:[ 1; 1 ] node "dilation" in
  (match hw2 esc `Dilation dilation with
  | 1, 1 -> ()
  | _ -> malformed esc (`Unsupported_option { op; option = `Dilation dilation }));
  let ceil_mode = bool_arg esc ~default:false node "ceil_mode" in
  { Pool.MaxPool2d.ceil_mode; kernel; stride; pad }

(* [avg_pool2d.default] carries no [dilation] argument at all (unlike the
   max-pool overloads) but does carry [count_include_pad] (represented, see
   [Pool.AvgPool2d.params.count_include_pad]) and [divisor_override] (has no
   field to hold a non-default value, so a present one is refused). *)
let avg_pool_params esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kernel, stride, pad = pool_window_fields esc node in
  (match int_opt_arg_opt esc node "divisor_override" with
  | None -> ()
  | Some d ->
      malformed esc (`Unsupported_option { op; option = `Divisor_override d }));
  let ceil_mode = bool_arg esc ~default:false node "ceil_mode" in
  let count_include_pad = bool_arg esc ~default:true node "count_include_pad" in
  { Pool.AvgPool2d.ceil_mode; count_include_pad; kernel; stride; pad }
