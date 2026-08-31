(* The Native4D operation: a CLOSED variant, separate from [Graph_ir.op].

   .ai/native4d_design.md §3 is emphatic about why this is not a flag on the
   Native op: "Making it a flag on [Graph_ir.op] would allow mixed or invalid
   graphs and would weaken exhaustiveness at every operation dispatch." A
   dialect whose illegal states are unrepresentable is the whole value of the
   exercise, and a shared variant gives that up on the first line.

   Two absences are decisions, not omissions:

   - NO [Discard]. Dead outputs are gone before conversion —
     [Pipeline.canonical] removes the sinks — so the dialect needs no way to
     mark an edge unused. [Discard] is the one Native op handled INLINE at every
     [op_registry] fold site (operands, map_operands, pp, JSON), so not
     inheriting it is what keeps the registry below free of special cases.
     (This used to be argued from "no Native4D op is multi-output". [Unbind]
     ends that, and the conclusion is unaffected: multi-output and zero-output
     are different needs, and only the second is what [Discard] serves.)
   - NO transposed grouped convolution. One group is [Transposed_conv2d];
     any other count is rejected at the domain check. A forward grouped
     convolution instead gets its own constructor, [Grouped_conv2d]
     (`.ai/native4d_design.md` §8's "smallest honest extension"), since
     [groups] there is a genuine parameter rather than a value a graph could
     misuse to claim [Conv2d] or [Depthwise_conv2d] semantics it does not have.

   Constructors in global alphabetical order, per the repo rule. *)

type op =
  | Add of Pointwise.Add.t
  | Add_scalar of Pointwise.Add_scalar.t
  | Adaptive_avg_pool2d of Pool.AdaptiveAvgPool2d.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Batch_norm_no_stats of Ops4.Batch_norm_no_stats.t
  | Clamp of Pointwise.Clamp.t
  | Concat4 of Ops4.Concat4.t
  | Conv2d of Ops4.Conv2d.t
  | Depthwise_conv2d of Ops4.Depthwise_conv2d.t
  | Div of Pointwise.Div.t
  | Div_scalar of Pointwise.Div_scalar.t
  | Expand4 of Ops4.Expand4.t
  | Gelu of Pointwise.Gelu.t
  | Group_norm4 of Ops4.Group_norm4.t
  | Grouped_conv2d of Ops4.Grouped_conv2d.t
  | Hardsigmoid of Pointwise.Hardsigmoid.t
  | Hardswish of Pointwise.Hardswish.t
  | Hardtanh of Pointwise.Hardtanh.t
  | Layer_norm of Ops4.Layer_norm.t
  | Leaky_relu of Pointwise.Leaky_relu.t
  | Max_keepdims of Ops4.Max_keepdims.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Mean_keepdims of Ops4.Mean_keepdims.t
  | Mul of Pointwise.Mul.t
  | Mul_scalar of Pointwise.Mul_scalar.t
  | Pad4 of Ops4.Pad4.t
  | Permute4 of Ops4.Permute4.t
  | Pow of Pointwise.Pow.t
  | Relu of Pointwise.Relu.t
  | Repeat4 of Ops4.Repeat4.t
  | RepeatInterleave4 of Ops4.RepeatInterleave4.t
  | Reshape4 of Ops4.Reshape4.t
  | Rms_norm of Ops4.Rms_norm.t
  | Rsub_scalar of Pointwise.Rsub_scalar.t
  | Select4 of Ops4.Select4.t
  | Select_scatter4 of Ops4.Select_scatter4.t
  | Sigmoid of Pointwise.Sigmoid.t
  | Silu of Pointwise.Silu.t
  | Slice4 of Ops4.Slice4.t
  | Split_with_sizes4 of Ops4.Split_with_sizes4.t
  | Sqrt of Pointwise.Sqrt.t
  | Stack4 of Ops4.Stack4.t
  | Sub of Pointwise.Sub.t
  | Sum_keepdims of Ops4.Sum_keepdims.t
  | To_copy of Pointwise.To_copy.t
  | Transposed_conv2d of Ops4.Transposed_conv2d.t
  | Unbind of Ops4.Unbind.t
  | Upsample_bilinear2d of Resize.Bilinear2d.t
  | Upsample_nearest2d of Resize.Nearest2d.t
  | Vector_norm_keepdims of Ops4.Vector_norm_keepdims.t
  | Arange4 of Ops4.Arange4.t
  | Zeros4 of Ops4.Zeros4.t

type t = op

(* The same per-op interface [Graph_ir] uses, for the same reason: the shared
   serialise / dataflow / pp logic folds a registry of payload modules instead
   of matching every constructor, so adding an op is one module plus one entry.
   Eleven of the nineteen entries are Native's payload modules UNCHANGED —
   their parameters name no axis and carry no shape, so there is nothing
   four-axis about them to restate. *)
module type OP = sig
  type t

  val name : string
  val jsont : t Jsont.t
  val operands : t -> Tensor_ref.t list
  val map_operands : (Tensor_ref.t -> Tensor_ref.t) -> t -> t
  val pp : Tensor_ref.t Fmt.t -> Format.formatter -> t -> unit
  val inject : t -> op
  val project : op -> t option
end

let op_registry : (module OP) list =
  [
    (module struct
      include Pointwise.Add

      let inject t = Add t
      let project = function Add t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Add_scalar

      let inject t = Add_scalar t
      let project = function Add_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AdaptiveAvgPool2d

      let inject t = Adaptive_avg_pool2d t
      let project = function Adaptive_avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AvgPool2d

      let inject t = Avg_pool2d t
      let project = function Avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Batch_norm_no_stats

      let inject t = Batch_norm_no_stats t
      let project = function Batch_norm_no_stats t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Clamp

      let inject t = Clamp t
      let project = function Clamp t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Concat4

      let inject t = Concat4 t
      let project = function Concat4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Conv2d

      let inject t = Conv2d t
      let project = function Conv2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Depthwise_conv2d

      let inject t = Depthwise_conv2d t
      let project = function Depthwise_conv2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Div

      let inject t = Div t
      let project = function Div t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Div_scalar

      let inject t = Div_scalar t
      let project = function Div_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Expand4

      let inject t = Expand4 t
      let project = function Expand4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Gelu

      let inject t = Gelu t
      let project = function Gelu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Group_norm4

      let inject t = Group_norm4 t
      let project = function Group_norm4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Grouped_conv2d

      let inject t = Grouped_conv2d t
      let project = function Grouped_conv2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardsigmoid

      let inject t = Hardsigmoid t
      let project = function Hardsigmoid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardswish

      let inject t = Hardswish t
      let project = function Hardswish t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardtanh

      let inject t = Hardtanh t
      let project = function Hardtanh t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Layer_norm

      let inject t = Layer_norm t
      let project = function Layer_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Leaky_relu

      let inject t = Leaky_relu t
      let project = function Leaky_relu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Max_keepdims

      let inject t = Max_keepdims t
      let project = function Max_keepdims t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.MaxPool2d

      let inject t = Max_pool2d t
      let project = function Max_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Mean_keepdims

      let inject t = Mean_keepdims t
      let project = function Mean_keepdims t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Mul

      let inject t = Mul t
      let project = function Mul t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Mul_scalar

      let inject t = Mul_scalar t
      let project = function Mul_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Pad4

      let inject t = Pad4 t
      let project = function Pad4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Permute4

      let inject t = Permute4 t
      let project = function Permute4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Pow

      let inject t = Pow t
      let project = function Pow t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Relu

      let inject t = Relu t
      let project = function Relu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Repeat4

      let inject t = Repeat4 t
      let project = function Repeat4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.RepeatInterleave4

      let inject t = RepeatInterleave4 t
      let project = function RepeatInterleave4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Reshape4

      let inject t = Reshape4 t
      let project = function Reshape4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Rms_norm

      let inject t = Rms_norm t
      let project = function Rms_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Rsub_scalar

      let inject t = Rsub_scalar t
      let project = function Rsub_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Select4

      let inject t = Select4 t
      let project = function Select4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Select_scatter4

      let inject t = Select_scatter4 t
      let project = function Select_scatter4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sigmoid

      let inject t = Sigmoid t
      let project = function Sigmoid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Silu

      let inject t = Silu t
      let project = function Silu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Slice4

      let inject t = Slice4 t
      let project = function Slice4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Split_with_sizes4

      let inject t = Split_with_sizes4 t
      let project = function Split_with_sizes4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sqrt

      let inject t = Sqrt t
      let project = function Sqrt t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Stack4

      let inject t = Stack4 t
      let project = function Stack4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sub

      let inject t = Sub t
      let project = function Sub t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Sum_keepdims

      let inject t = Sum_keepdims t
      let project = function Sum_keepdims t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.To_copy

      let inject t = To_copy t
      let project = function To_copy t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Transposed_conv2d

      let inject t = Transposed_conv2d t
      let project = function Transposed_conv2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Unbind

      let inject t = Unbind t
      let project = function Unbind t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Bilinear2d

      let inject t = Upsample_bilinear2d t
      let project = function Upsample_bilinear2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Nearest2d

      let inject t = Upsample_nearest2d t
      let project = function Upsample_nearest2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Vector_norm_keepdims

      let inject t = Vector_norm_keepdims t
      let project = function Vector_norm_keepdims t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Arange4

      let inject t = Arange4 t
      let project = function Arange4 t -> Some t | _ -> None
    end : OP);
    (module struct
      include Ops4.Zeros4

      let inject t = Zeros4 t
      let project = function Zeros4 t -> Some t | _ -> None
    end : OP);
  ]

let registry_pick (f : (module OP) -> 'a option) : 'a =
  Option.get (List.find_map f op_registry)

(* No [Discard] arm to write first, unlike [Graph_ir]'s: every op is in the
   registry. *)
let operands (op : op) =
  registry_pick (fun (module M : OP) -> Option.map M.operands (M.project op))

let map_operands f (op : op) =
  registry_pick (fun (module M : OP) ->
      Option.map (fun t -> M.inject (M.map_operands f t)) (M.project op))

(* The registry's own case tag, for reports that count ops by kind. *)
let name (op : op) =
  registry_pick (fun (module M : OP) ->
      Option.map (fun _ -> M.name) (M.project op))

let pp_with ~(pp_ref : Tensor_ref.t Fmt.t) fmt (op : op) =
  registry_pick (fun (module M : OP) ->
      Option.map (fun t -> M.pp pp_ref fmt t) (M.project op))

let pp fmt op = pp_with ~pp_ref:Tensor_id.pp fmt op

(* One [(case, decoder)] per op, keyed by [name], through the same
   [Json_util.union]/[single] pair [Graph_ir] uses — and with no [Discard] case
   to prepend, which is the one place that special case shows up there. *)
let op_decode_cases : (string * (Jsont.json -> op)) list =
  List.map
    (fun (module M : OP) ->
      (M.name, fun v -> M.inject (Json_util.dec M.jsont v)))
    op_registry

let jsont : op Jsont.t =
  Jsont.map ~kind:"native4d_op"
    ~dec:(fun json -> Json_util.union ~kind:"native4d_op" op_decode_cases json)
    ~enc:(fun op ->
      registry_pick (fun (module M : OP) ->
          Option.map
            (fun t -> Json_util.single ~case:M.name (Json_util.enc M.jsont t))
            (M.project op)))
    Jsont.json
