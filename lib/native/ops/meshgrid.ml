(* [meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]],
   restricted to the corpus's own shape: every input rank-1, [indexing="ij"]
   (real ATen's "xy" convention swaps the first two OUTPUT axes for every
   output, a different rule with no corpus evidence -- rejected rather than
   guessed at, the same discipline [Pointwise.To_copy]'s three-way domain
   documents for itself).

   Output count equals input count, one axis per input -- rank N, right-
   aligned via [Aten_shape.used_axes] the same way every other rank-derived
   op resolves its axes. Every output shares the SAME shape (all N axes, one
   per input's own extent); output k differs only in VALUE: it reads ONLY
   input k, broadcast along every other axis -- no arithmetic, a pure
   coordinate remap from output axis k to input k's own single real axis
   (C, since each input is rank-1 -- [Aten_shape.of_aten]'s own encoding). *)
module Meshgrid = struct
  type t = { tensors : Tensor_ref.t list }

  let name = "Meshgrid"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          tensors =
            Json_util.req_field ms "tensors" (Jsont.list Tensor_ref.jsont) name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ( "tensors",
              Json_util.jarr
                (List.map (Json_util.enc Tensor_ref.jsont) t.tensors) );
          ])
      Jsont.json

  let operands (t : t) = t.tensors
  let map_operands f (t : t) = { tensors = List.map f t.tensors }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>meshgrid@ tensors=%a@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_ref))
      t.tensors

  (* Every input must be rank-1 (extent on [C] alone, every outer axis 1 --
     [Aten_shape.of_aten]'s own rank-1 encoding); anything else has no Native
     value representation as a meshgrid axis and is rejected. *)
  let check_rank1 (shape : Vec6.shape) =
    let ones_outside_c =
      List.for_all
        (fun ax ->
          Axis.equal ax Axis.C || Dim.equal (Vec6.get shape ax) Dim.one)
        Axis.all
    in
    if ones_outside_c then Err.return () else Err.fail (`Meshgrid shape)

  let output_shapes (shapes : Vec6.shape list) =
    let open Err.Syntax in
    let* () = Err.List.iter check_rank1 shapes in
    let axes = Aten_shape.used_axes ~rank:(Rank.of_list shapes) in
    let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
    let out_shape =
      List.fold_left2
        (fun s ax shape -> Vec6.copy shape ~src:Axis.C ~dst:ax s)
        ones axes shapes
    in
    Err.return (List.map (fun _ -> out_shape) shapes)

  module Compute (S : Semantics.SEMANTICS) = struct
    let zero =
      Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero ~h:S.index_zero
        ~w:S.index_zero ~c:S.index_zero

    (* [axis] is the axis this output's ordinal owns, resolved by the
       caller (its position among [used_axes ~rank:(Rank.of_list xs)]).
       Reads [x] (rank-1, real data on [C]) at the coordinate [out] carries
       on [axis], moved onto [C] -- the inverse of the shape rule's own
       [Vec6.copy ~src:C ~dst:axis]. *)
    let pixel ~axis x (out : Semantics.position S.index Vec6.t) =
      S.load x (Vec6.copy out ~src:axis ~dst:Axis.C zero)
  end
end
