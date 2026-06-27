(* A tensor input described compactly for verification: its dtype and shape plus
   how to fill it. Payloads are usually *synthesized* (Random/Sequence) rather
   than stored; [Values] holds explicit, bit-exact contents when needed.

   The payload is a single key alongside dtype/shape, so a tensor reads as
   { "dtype": "f32", "shape": [2,3], "random": { "uniform": {...} } }. *)

type seq = { start : float; step : float }

type source =
  | Values of Scalar_value.t list
  | Random of Distribution.t
  | Sequence of seq

type t = { dtype : Dtype.t; shape : int list; source : source }

let seq_payload : seq Jsont.t =
  Jsont.Object.map ~kind:"sequence" (fun start step -> { start; step })
  |> Jsont.Object.mem "start" Jsont.number
  |> Jsont.Object.mem "step" Jsont.number
  |> Jsont.Object.finish

let jsont : t Jsont.t =
  Jsont.map ~kind:"tensor"
    ~dec:(fun json ->
      match Spec_util.members json with
      | None -> Jsont.Error.msgf Jsont.Meta.none "tensor: expected an object"
      | Some ms ->
          let dtype =
            match Spec_util.find_member ms "dtype" with
            | Some v -> Spec_util.dec Dtype.jsont v
            | None ->
                Jsont.Error.msgf Jsont.Meta.none "tensor: missing \"dtype\""
          in
          let shape =
            match Spec_util.find_member ms "shape" with
            | Some v -> Spec_util.dec (Jsont.list Jsont.int) v
            | None ->
                Jsont.Error.msgf Jsont.Meta.none "tensor: missing \"shape\""
          in
          let source =
            match
              ( Spec_util.find_member ms "values",
                Spec_util.find_member ms "random",
                Spec_util.find_member ms "sequence" )
            with
            | Some v, None, None ->
                Values (Spec_util.dec (Jsont.list Scalar_value.jsont) v)
            | None, Some v, None -> Random (Spec_util.dec Distribution.jsont v)
            | None, None, Some v -> Sequence (Spec_util.dec seq_payload v)
            | None, None, None ->
                Jsont.Error.msgf Jsont.Meta.none
                  "tensor: needs one of \"values\"/\"random\"/\"sequence\""
            | _ ->
                Jsont.Error.msgf Jsont.Meta.none
                  "tensor: at most one payload source"
          in
          { dtype; shape; source })
    ~enc:(fun t ->
      let source_kv =
        match t.source with
        | Values vs ->
            ( "values",
              Spec_util.jarr (List.map (Spec_util.enc Scalar_value.jsont) vs) )
        | Random d -> ("random", Spec_util.enc Distribution.jsont d)
        | Sequence { start; step } ->
            ( "sequence",
              Spec_util.jobj
                [
                  ("start", Spec_util.jnum start); ("step", Spec_util.jnum step);
                ] )
      in
      Spec_util.jobj
        [
          ("dtype", Spec_util.enc Dtype.jsont t.dtype);
          ("shape", Spec_util.jarr (List.map Spec_util.jint t.shape));
          source_kv;
        ])
    Jsont.json
