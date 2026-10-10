(* One Native node for aten.diff.default. The admitted slice is n=0/1,
   same-dtype F32/I64 inputs and nonempty results. Optional operands are
   virtual concatenation segments; their loads are clamped because a
   symbolic select evaluates both branches. Integer data never enters the
   float domain. Higher orders need intermediate binary32 materialization. *)
module Diff = struct
  type params = { axis : Axis.t; order : Aten_int.Size.t }

  type t = {
    params : params;
    x : Tensor_ref.t;
    prepend : Tensor_ref.t option;
    append : Tensor_ref.t option;
  }

  let name = "Diff"

  let params_jsont =
    Jsont.Object.map ~kind:"diff_params" (fun axis order -> { axis; order })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "order"
         (Jsont.map ~dec:Aten_int.Size.of_int
            ~enc:(fun n -> (n :> int))
            Jsont.int)
         ~enc:(fun p -> p.order)
    |> Jsont.Object.finish

  let jsont : t Jsont.t =
    Jsont.Object.map ~kind:name (fun params x prepend append ->
        { params; x; prepend; append })
    |> Jsont.Object.mem "params" params_jsont ~enc:(fun t -> t.params)
    |> Jsont.Object.mem "x" Tensor_ref.jsont ~enc:(fun t -> t.x)
    |> Jsont.Object.opt_mem "prepend" Tensor_ref.jsont ~enc:(fun t -> t.prepend)
    |> Jsont.Object.opt_mem "append" Tensor_ref.jsont ~enc:(fun t -> t.append)
    |> Jsont.Object.finish

  (* Preserve schema operand order: self, prepend, append. *)
  let operands (t : t) = t.x :: List.filter_map Fun.id [ t.prepend; t.append ]

  let map_operands f (t : t) =
    {
      t with
      x = f t.x;
      prepend = Option.map f t.prepend;
      append = Option.map f t.append;
    }

  let pp pp_ref fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>diff@ x=%a@ axis=%a@ order=%a@ prepend=%a@ append=%a@]"
      pp_ref t.x Axis.pp t.params.axis Aten_int.Size.pp t.params.order
      (Fmt.option pp_ref) t.prepend (Fmt.option pp_ref) t.append

  (* Segment order is the semantic concatenation order, not alphabetical. *)
  let segments ~x ~prepend ~append =
    Option.to_list prepend @ [ x ] @ Option.to_list append

  let output_shape (p : params) ~(x : Tensor_sig.t) ~prepend ~append =
    let open Err.Syntax in
    let order = (p.order :> int) in
    if order < 0 || order > 1 then
      Err.fail (`Diff (Shape_error.Diff.Unsupported_order p.order))
    else
      let all = segments ~x ~prepend ~append in
      let* () =
        match x.Tensor_sig.fmt with
        | Payload.(Fmt F32 | Fmt I64) ->
            Err.List.iter
              (fun (sg : Tensor_sig.t) ->
                match (order, x.fmt, sg.fmt) with
                | 0, _, _ -> Err.return ()
                | _, Payload.Fmt Payload.F32, Payload.Fmt Payload.F32
                | _, Payload.Fmt Payload.I64, Payload.Fmt Payload.I64 ->
                    Err.return ()
                | _ -> Err.fail (`Diff (Shape_error.Diff.Mixed_dtype sg.fmt)))
              all
        | fmt -> Err.fail (`Diff (Shape_error.Diff.Unsupported_dtype fmt))
      in
      let* () =
        Err.List.iter
          (fun (sg : Tensor_sig.t) ->
            Err.List.iter
              (fun axis ->
                let first = Vec6.get x.shape axis
                and other = Vec6.get sg.shape axis in
                if Axis.equal axis p.axis || Dim.equal first other then
                  Err.return ()
                else
                  Err.fail
                    (`Concat
                       (Shape_error.Concat.Axis_mismatch { axis; first; other })))
              Axis.all)
          all
      in
      if order = 0 then Err.return x.shape
      else
        let* joined =
          Concat.Concat.output_shape
            Concat.Concat.{ axis = p.axis }
            ~xs_shapes:(List.map (fun (sg : Tensor_sig.t) -> sg.shape) all)
        in
        let count = (Vec6.get joined p.axis :> int) - 1 in
        if count = 0 then Err.fail (`Diff Shape_error.Diff.Empty_result)
        else Err.return (Vec6.set joined p.axis (Dim.extent count))

  module Read
      (S : Semantics.SEMANTICS)
      (V : sig
        type value

        val load : S.input -> Semantics.position S.index Vec6.t -> value
        val select : S.b -> value -> value -> value
        val sub : value -> value -> value
      end) =
  struct
    let rec read axis out = function
      | [] -> invalid_arg "Diff.read: no segments"
      | ((extent : Dim.extent Dim.t), input) :: rest -> (
          let j = S.of_index (Vec6.get out axis) in
          let src =
            S.clamp_low (S.index_min (S.index_const ((extent :> int) - 1)) j)
          in
          let value = V.load input (Vec6.set out axis src) in
          match rest with
          | [] -> value
          | _ ->
              let next =
                S.clamp_low (S.index_add j (S.index_const (-(extent :> int))))
              in
              V.select
                (S.index_eq j (S.of_index src))
                value
                (read axis (Vec6.set out axis next) rest))

    let pixel (p : params) ~x ~prepend ~append out =
      if (p.order :> int) = 0 then V.load (snd x) out
      else
        let xs = segments ~x ~prepend ~append in
        let xs = List.map (fun (sh, input) -> (Vec6.get sh p.axis, input)) xs in
        let next =
          S.assume_index
            (S.index_add (S.of_index (Vec6.get out p.axis)) (S.index_const 1))
        in
        V.sub (read p.axis (Vec6.set out p.axis next) xs) (read p.axis out xs)
  end

  module Compute (S : Semantics.SEMANTICS) =
    Read
      (S)
      (struct
        type value = S.t

        let load = S.load
        let select = S.select
        let sub = S.sub
      end)

  module Compute_i64
      (S : Semantics.SEMANTICS)
      (T : sig
        type 'a repr

        val i64_load :
          S.input -> Semantics.position S.index Vec6.t -> int64 repr

        val i64_binary :
          Expr.Value.i64_binary_op -> int64 repr -> int64 repr -> int64 repr

        val typed_select : S.b -> 'a repr -> 'a repr -> 'a repr
      end) =
    Read
      (S)
      (struct
        type value = int64 T.repr

        let load = T.i64_load
        let select = T.typed_select
        let sub = T.i64_binary Expr.Value.I64_sub
      end)
end
