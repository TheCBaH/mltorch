(* `embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
   bool scale_grad_by_freq=False, bool sparse=False) -> Tensor`: row lookup,
   [out[..., :] = weight[indices[...], :]].

   Its own node, not a rewrite of [index.Tensor], because the source operation
   is different and so is its failure behavior: an embedding lookup is
   ATen's [index_select], which rejects every index outside [0, V), whereas
   [index.Tensor] is advanced indexing and accepts [-V, -1] by wrapping. The
   evaluation delegates to [Index_tensor] after translating the parameters --
   a lookup along the weight's vocabulary axis with an index of
   [indices_rank] real axes -- so the gather, its shape rule and its symbolic
   form are the ones that op already has. Direct evaluation adds the strict
   check ([Eval_direct_compute]); a wrapped negative index is therefore
   reachable only through the symbolic/generated routes, which share
   [index.Tensor]'s gather, and is recorded as such in the design record.

   [padding_idx] marks a row that ATen's BACKWARD pass leaves untouched. The
   forward lookup ignores it: the stored padding row is returned as stored,
   never zeroed. It is kept in the payload for provenance and round-trips; it
   changes no value. [scale_grad_by_freq] and [sparse] are gradient options
   with no forward effect, so the importers read and discard them, and nothing
   here implies training support. *)

module Embedding = struct
  (* An index outside [0, vocabulary): ATen's [index_select] rejects it, and so
     does Direct evaluation. *)
  module Index_out_of_range = struct
    type t = { raw : int64; vocabulary : Dim.extent Dim.t }

    let pp fmt { raw; vocabulary } =
      Fmt.pf fmt "embedding index %Ld out of range [0, %a)" raw Dim.pp
        vocabulary
  end

  type params = { indices_rank : Rank.t; padding_idx : Aten_int.Index.t }

  let params_jsont : params Jsont.t =
    let index =
      Jsont.map ~kind:"padding_idx" ~dec:Aten_int.Index.of_int
        ~enc:Aten_int.Index.to_int Jsont.int
    in
    Jsont.Object.map ~kind:"embedding_params" (fun indices_rank padding_idx ->
        { indices_rank; padding_idx })
    |> Jsont.Object.mem "indices_rank" Rank.jsont ~enc:(fun p -> p.indices_rank)
    |> Jsont.Object.mem "padding_idx" index ~enc:(fun p -> p.padding_idx)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{indices_rank=%a padding_idx=%a}@]" Rank.pp p.indices_rank
      Aten_int.Index.pp p.padding_idx

  type t = { params : params; weight : Tensor_ref.t; indices : Tensor_ref.t }

  let name = "Embedding"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          weight = get "weight" Tensor_ref.jsont;
          indices = get "indices" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("weight", Json_util.enc Tensor_ref.jsont t.weight);
            ("indices", Json_util.enc Tensor_ref.jsont t.indices);
          ])
      Jsont.json

  let operands (t : t) = [ t.weight; t.indices ]

  let map_operands f (t : t) =
    { t with weight = f t.weight; indices = f t.indices }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>embedding@ weight=%a@ indices=%a@ params=%a@]" pp_ref
      t.weight pp_ref t.indices pp_params t.params

  (* The lookup runs along the weight's vocabulary axis, W. *)
  let gather_params (p : params) : Index_tensor.Index_tensor.params =
    { axis = Axis.W; index_rank = p.indices_rank }

  (* Only W (the vocabulary) and C (the dimension) of the weight may exceed one;
     the rest is [Index_tensor]'s: that [indices] really has [indices_rank]
     axes, and that the output axes it borrows are free. *)
  let output_shape ~(weight_shape : Vec6.shape) ~(indices_shape : Vec6.shape)
      (p : params) =
    let open Err.Syntax in
    let* () =
      if
        List.for_all
          (fun a -> Dim.equal (Vec6.get weight_shape a) Dim.one)
          [ Axis.N; Axis.T; Axis.D; Axis.H ]
      then Err.return ()
      else
        Err.fail
          (`Embedding (Shape_error.Embedding.Weight_not_a_matrix weight_shape))
    in
    Index_tensor.Index_tensor.output_shape ~self_shape:weight_shape
      ~index_shape:indices_shape (gather_params p)

  module Compute (S : Semantics.SEMANTICS) = struct
    module G = Index_tensor.Index_tensor.Compute (S)

    let pixel (p : params) ~(weight_shape : Vec6.shape) ~weight ~indices
        (out : Semantics.position S.index Vec6.t) =
      G.pixel (gather_params p) ~self_shape:weight_shape ~self:weight
        ~index:indices out
  end
end
