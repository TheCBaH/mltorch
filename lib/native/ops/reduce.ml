(* Reductions over a set of axes (NHWC). Currently just [Mean] (ATen's
   `aten.mean.dim`); a category file so future `aten.sum`/`amax`/… land beside
   it, the way [Pool] holds Max/AvgPool2d. *)

module Mean = struct
  (* Arithmetic mean over [dims]: each reduced axis is summed over its full
     extent, divided by their product. resnet18's global average pool is this
     over H/W — [AvgPool2d] with the window spanning the whole axis, but
     expressed directly as a reduction (no kernel/stride/pad/clipping: the full
     axis is always in range). See .ai/native_compute_design.md §2.

     [dims] is a list of frame axes (a dispatcher gets them from an ATen dim list
     via [Aten_shape.axis_of_dim]); one op covers `mean(dim=[H,W])`.

     [keepdim] mirrors `aten.mean.dim`'s flag (.ai/native_tensor_design.md §1d):
       - keepdim=true  → each reduced axis collapses to extent 1 *in place*; the
         named layout is otherwise unchanged.
       - keepdim=false → the reduced axes are removed and the survivors re-pack
         toward the inside, so the named axis carrying a given input's data
         shifts. mean over W of [H6 W7 C8] gives [H1 W6 C8] — H's data now lives
         on W (whereas keepdim=true gives [H6 W1 C8]).
     The op stays a rank-free function of named axes: the keepdim=false shift
     needs no input rank, because under the right-aligned embedding the axes
     outside the data are already size 1, so shifting *every* surviving axis
     (real or size-1 padding) inward by the number of removed axes outer to it
     lands the real data on exactly the right-aligned axes. (Recovering the ATen
     output *rank* is a separate concern — [Aten_shape.to_aten], not the op's.) *)
  type params = { dims : Axis.t list; keepdim : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"mean_params" (fun dims keepdim -> { dims; keepdim })
    |> Jsont.Object.mem "dims" (Jsont.list Axis.jsont) ~enc:(fun p -> p.dims)
    |> Jsont.Object.mem "keepdim" Jsont.bool ~enc:(fun p -> p.keepdim)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%a}@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Axis.pp))
      p.dims Fmt.bool p.keepdim

  type t = { params : params; x : Tensor_ref.t }

  let name = "Mean"

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
    Fmt.pf fmt "@[<hv 2>mean@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* Maps each surviving INPUT axis to the OUTPUT axis that carries its data.
     keepdim=true: identity (collapse in place). keepdim=false: the survivors
     (all axes minus the reduced ones, in canonical order) re-pack right-aligned
     into the innermost positions. Depends only on [p], not on extents. *)
  let kept_map (p : params) =
    let survivors = List.filter (fun a -> not (List.mem a p.dims)) Axis.all in
    if p.keepdim then List.map (fun a -> (a, a)) survivors
    else
      List.combine survivors
        (Aten_shape.used_axes ~rank:(List.length survivors))

  (* Output = surviving input extents packed onto their output axes (§1d);
     reduced axes (and, for keepdim=false, the vacated outer axes) are extent 1. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
    List.fold_left
      (fun s (kin, oax) -> Vec6.set s oax (Vec6.get x_shape kin))
      ones (kept_map p)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x
        (out : Axis.t -> Semantics.position S.index) =
      let kmap = kept_map p in
      (* Nest one [sum] per reduced axis (full extent [0, extent)); the body
         reads [x], mapping each input axis: reduced → its reduction variable,
         surviving → the output coord at the axis it re-packed onto. *)
      let rec reduce dims override =
        match dims with
        | [] ->
            S.load x (fun a ->
                match List.assoc_opt a override with
                | Some i -> i
                | None -> (
                    match List.assoc_opt a kmap with
                    | Some oax -> out oax
                    | None -> S.index_zero))
        | d :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> reduce rest ((d, i) :: override))
      in
      let total = reduce p.dims [] in
      let count =
        List.fold_left (fun acc d -> acc * (Vec6.get x_shape d :> int)) 1 p.dims
      in
      S.div total (S.const (float_of_int count))
  end
end
