(* Split out of conv.ml see conv.ml (the facade). Depends on
   Conv2d for its axis_window/params types and Compute functor. *)

open Conv_conv2d

module Conv2d_padding = struct
  type padding = Same | Valid

  (* The CHECKED form, and the one every importer must use: the mode arrives as
     model data, so an unknown one is a fact about the graph and not a broken
     precondition. Returns the offered string, which is all a caller needs to
     report it -- and both importers report it in their own row rather than
     sharing one, since the two error domains are separate. *)
  let of_string = function
    | "same" -> Ok Same
    | "valid" -> Ok Valid
    | s -> Error s

  (* Asserting, for the Jsont decoder alone: a mode read back out of a graph the
     engine itself wrote is a trusted value. *)
  let padding_of_string s =
    match of_string s with
    | Ok p -> p
    | Error s -> invalid_arg ("Conv2d_padding: invalid padding " ^ s)

  let string_of_padding = function Same -> "same" | Valid -> "valid"

  let padding_jsont : padding Jsont.t =
    Jsont.map ~kind:"conv2d_padding_mode"
      ~dec:(fun s ->
        try padding_of_string s
        with Invalid_argument msg -> Jsont.Error.msgf Jsont.Meta.none "%s" msg)
      ~enc:string_of_padding Jsont.string

  let pp_padding fmt p = Fmt.string fmt (string_of_padding p)

  type params = {
    stride : Op_config.Pos.t Op_config.Hw.t;
    padding : padding;
    dilation : Op_config.Pos.t Op_config.Hw.t;
    groups : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"conv2d_padding_params"
      (fun stride padding dilation groups ->
        { stride; padding; dilation; groups })
    |> Jsont.Object.mem "stride" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.stride)
    |> Jsont.Object.mem "padding" padding_jsont ~enc:(fun p -> p.padding)
    |> Jsont.Object.mem "dilation" (Op_config.Hw.jsont Op_config.Pos.jsont)
         ~enc:(fun p -> p.dilation)
    |> Jsont.Object.mem "groups" Op_config.Pos.jsont ~enc:(fun p -> p.groups)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{stride=%a;@ padding=%a;@ dilation=%a;@ groups=%a}@]"
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.stride pp_padding p.padding
      (Op_config.Hw.pp Op_config.Pos.pp)
      p.dilation Op_config.Pos.pp p.groups

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t;
    bias : Tensor_ref.t option;
  }

  let name = "Conv2d_padding"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = get "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt_bias =
          match t.bias with None -> [] | Some r -> [ ("bias", ref_ r) ]
        in
        Json_util.jobj
          (opt_bias
          @ [
              ("params", Json_util.enc params_jsont t.params);
              ("weight", ref_ t.weight);
              ("x", ref_ t.x);
            ]))
      Jsont.json

  let operands (t : t) = [ t.x; t.weight ] @ Option.to_list t.bias

  let map_operands f (t : t) =
    { t with x = f t.x; weight = f t.weight; bias = Option.map f t.bias }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>conv2d_padding@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x pp_ref t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  (* [total] is a product of two independently model-supplied factors, so it is
     computed in [int64] and bounded before narrowing -- the same rule
     [Window_axis.output_extent] follows, and for the same reason: under
     js_of_ocaml [int] is 32 bits, and in [int] this can wrap to a negative
     number that then reaches [Op_config.Nonneg.of_int]'s assertion as an
     escaping [Invalid_argument] instead of a typed error. *)
  let same_padding ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
      ~(dilation : Op_config.Pos.t) =
    if (stride :> int) <> 1 then
      Err.fail
        (`Convolution
           (Shape_error.Convolution.Same_padding_requires_stride_one { stride }))
    else
      let open Err.Syntax in
      (* Each FACTOR bounded before the multiplication, not only the product:
         [Op_config.Pos] enforces sign and not magnitude, so on a 63-bit-[int]
         build [Int64.mul] of two unbounded factors wraps and lands back inside
         the range a post-hoc check accepts. Same rule and same helper as
         [Window_axis.output_extent]. *)
      let* d = Window_axis.factor ~what:`Dilation (dilation :> int) in
      let* k = Window_axis.factor ~what:`Kernel (kernel :> int) in
      let total = Int64.mul d (Int64.sub k 1L) in
      if total >= Window_axis.limit then
        Err.fail
          (`Window_over_limit
             Shape_error.Window_over_limit.
               {
                 what = `Effective_kernel;
                 value = Int64.add total 1L;
                 limit = Window_axis.limit;
               })
      else
        let total = Int64.to_int total in
        Err.return
          ( Op_config.Nonneg.of_int (total / 2),
            Op_config.Nonneg.of_int (total - (total / 2)) )

  let axis_window ~(padding : padding) ~(kernel : Dim.extent Dim.t)
      ~(stride : Op_config.Pos.t) ~(dilation : Op_config.Pos.t) :
      (Conv2d.axis_window, Shape_error.t) Err.t =
    let open Err.Syntax in
    let* pad_before, pad_after =
      match padding with
      | Same -> same_padding ~kernel ~stride ~dilation
      | Valid ->
          Err.return (Op_config.Nonneg.of_int 0, Op_config.Nonneg.of_int 0)
    in
    Err.return { Conv2d.kernel; stride; pad_before; pad_after; dilation }

  let to_conv2d_params ~(weight_shape : Vec6.shape) (p : params) :
      (Conv2d.params, Shape_error.t) Err.t =
    let open Err.Syntax in
    (* Same aggregate rule as [Native_interp.conv_in_channels], which computes
       this from serialized metadata: the per-group input extent times the group
       count is a product of two model-supplied factors and can exceed the
       engine's per-axis ceiling even when both factors are inside it. *)
    let* in_channels =
      Window_axis.in_channels
        ~per_group:(Vec6.get weight_shape Axis.C)
        ~groups:p.groups
    in
    let* h =
      axis_window ~padding:p.padding
        ~kernel:(Vec6.get weight_shape Axis.H)
        ~stride:p.stride.h ~dilation:p.dilation.h
    in
    let+ w =
      axis_window ~padding:p.padding
        ~kernel:(Vec6.get weight_shape Axis.W)
        ~stride:p.stride.w ~dilation:p.dilation.w
    in
    { Conv2d.h; w; in_channels; groups = p.groups }

  let output_shape ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
      (p : params) =
    let open Err.Syntax in
    let* p = to_conv2d_params ~weight_shape p in
    Conv2d.output_shape ~x_shape ~weight_shape p

  module Compute (S : Semantics.SEMANTICS) = struct
    module C = Conv2d.Compute (S)

    let pixel (p : params) ~(x_shape : Vec6.shape) ~(weight_shape : Vec6.shape)
        ~x ~weight ~bias out =
      C.pixel
        (or_invalid_arg (to_conv2d_params ~weight_shape p))
        ~x_shape ~weight_shape ~x ~weight ~bias out
  end
end
