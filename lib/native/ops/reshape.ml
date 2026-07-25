(* Contiguous (row-major) reshape — the native twin of ATen's `view`/`reshape`
   on a contiguous tensor. Output element k (row-major over N,T,D,H,W,C in the
   target shape) is input element k (row-major in the input shape); i.e. it
   reinterprets the same flat buffer under a new shape. numel-preservation is a
   precondition (the bridge derives the target from ATen's own view, which is
   numel-preserving).

   The bridge wraps this in Permute nodes when the input isn't already in
   ATen-contiguous order; for `of_aten` inputs (right-aligned, contiguous) the
   native row-major order already equals ATen row-major, so none are needed. *)

module Reshape = struct
  type params = { shape : Vec6.shape }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"reshape_params" (fun shape -> { shape })
    |> Jsont.Object.mem "shape" Vec6.shape_jsont ~enc:(fun p -> p.shape)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{shape=%a}@]" Vec6.pp_shape p.shape

  type t = { params : params; x : Tensor_ref.t }

  let name = "Reshape"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" params_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>reshape@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  (* The output shape is exactly the target (numel-preserving precondition). *)
  let output_shape (p : params) = Core.return p.shape

  (* Walk config: just the input shape; the target flattens all elements onto C
     (always numel-preserving), which fully exercises the delinearize path. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 4 } }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let params (c : cfg) : params =
      let numel = (Vec6.numel (Walk_bridge.vec6 c.shape) :> int) in
      { shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:numel }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a -> flat}" Walk_core.Shape.pp c.shape
  end

  (* Row-major stride of each axis: the product of the extents of all axes
     strictly inner to it (C innermost). *)
  let strides (shape : Vec6.shape) : int Vec6.t =
    let ext a = (Vec6.get shape a :> int) in
    let rec go acc prod = function
      | [] -> acc
      | a :: rest -> go ((a, prod) :: acc) (prod * ext a) rest
    in
    let assoc = go [] 1 (List.rev Axis.all) in
    Vec6.mapi (fun a _ -> List.assoc a assoc) shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* row-major flat offset of [coord] under [shape]:
       off = ((((n*Tn+t)*Dn+d)*Hn+h)*Wn+w)*Cn+c *)
    let flat_offset (shape : Vec6.shape) coord =
      List.fold_left
        (fun acc a ->
          let e = (Vec6.get shape a :> int) in
          S.index_add (S.index_scale e acc) (S.of_index (Vec6.get coord a)))
        (S.index_const 0) Axis.all

    (* delinearize [off] into a coord under [shape]: per axis,
       c_a = (off / stride_a) mod extent_a, mod x d = x - d*(x/d). *)
    let delinearize (shape : Vec6.shape) off =
      let strides = strides shape in
      Vec6.mapi
        (fun a _ ->
          let stride = Op_config.Pos.of_int (Vec6.get strides a) in
          let ext = (Vec6.get shape a :> int) in
          let q = S.index_floor_div_pos off stride in
          let c =
            S.index_add q
              (S.index_scale (-ext)
                 (S.index_floor_div_pos q (Op_config.Pos.of_int ext)))
          in
          S.assume_index c)
        shape

    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Semantics.position S.index Vec6.t) =
      (* [out]'s row-major offset in the target shape is the same element's
         offset in the input shape; delinearize it there and load. *)
      let off = flat_offset p.shape out in
      S.load x (delinearize x_shape off)
  end
end
