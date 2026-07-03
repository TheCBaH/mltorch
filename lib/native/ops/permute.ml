(* Axis permutation: a full 6D rearrangement with data copy. Used by the bridge
   to relayout tensors between ATen's positional right-alignment and native
   semantic axis assignments (e.g. NCHW -> NHWC for conv/pool inputs/outputs).

   A permutation is an association list [(out_ax, in_ax)] covering all 6 axes
   exactly once on each side: output axis [out_ax] gets the extent and data from
   input axis [in_ax].  The type alias keeps call sites readable — the direction
   is always output-to-input. *)

module Permute = struct
  (* [(out_axis, in_axis)] — which input axis feeds each output axis. *)
  type perm = (Axis.t * Axis.t) list

  let perm_jsont : perm Jsont.t =
    Jsont.list
      (Jsont.Object.map ~kind:"axis_pair" (fun in_ax out_ax -> (out_ax, in_ax))
      |> Jsont.Object.mem "in" Axis.jsont ~enc:snd
      |> Jsont.Object.mem "out" Axis.jsont ~enc:fst
      |> Jsont.Object.finish)

  let pp_perm fmt (perm : perm) =
    let pp_axis_pair fmt (out_axis, in_axis) =
      Fmt.pf fmt "@[<h>%a<-%a@]" Axis.pp out_axis Axis.pp in_axis
    in
    Fmt.brackets
      (Fmt.list ~sep:Fmt.comma pp_axis_pair)
      fmt
      (List.filter
         (fun (out_axis, in_axis) -> not (Axis.equal out_axis in_axis))
         perm)

  type t = { perm : perm; x : Tensor_ref.t }

  let name = "Permute"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { perm = get "perm" perm_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("perm", Json_util.enc perm_jsont t.perm);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>permute@ x=%a@ perm=%a@]" pp_ref t.x pp_perm t.perm

  (* output_shape[out_ax] = x_shape[in_ax] for each (out_ax, in_ax) pair.
     Axes absent from [perm] default to extent 1 (never happens for a full perm). *)
  let output_shape ~(x_shape : Vec6.shape) (perm : perm) =
    let open Core.Syntax in
    let validate_axis side proj axis =
      let count =
        List.length (List.filter (fun pair -> Axis.equal (proj pair) axis) perm)
      in
      if count = 1 then Core.return ()
      else Core.fail (`Permute Shape_error.Permute.{ side; axis; count })
    in
    let* () =
      Core.List.fold_left
        (fun () axis -> validate_axis `Output fst axis)
        () Axis.all
    in
    let* () =
      Core.List.fold_left
        (fun () axis -> validate_axis `Input snd axis)
        () Axis.all
    in
    Core.return
      (List.fold_left
         (fun s (out_ax, in_ax) -> Vec6.copy x_shape ~src:in_ax ~dst:out_ax s)
         (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
         perm)

  module Compute (S : Semantics.SEMANTICS) = struct
    (* At each output coord [out], read the input at the coordinate produced by
       the inverse permutation: for each input axis [in_ax], use the output
       coordinate of the output axis that maps to it.
       Raises [Not_found] if [perm] is not a bijection over all 6 axes. *)
    let pixel perm ~x (out : Semantics.position S.index Vec6.t) =
      let inv = List.map (fun (oax, iax) -> (iax, oax)) perm in
      S.load x (Vec6.of_fn (fun in_ax -> Vec6.get out (List.assoc in_ax inv)))
  end
end
