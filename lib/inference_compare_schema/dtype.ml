module W = Wire
module Codec = W.Text (Id.Codec)

type t = {
  codec : Id.Codec.t;
  parameters_digest : Digest.Dtype_parameters.t option;
}

let compare a b =
  match Id.Codec.compare a.codec b.codec with
  | 0 ->
      Option.compare Digest.Dtype_parameters.compare a.parameters_digest
        b.parameters_digest
  | c -> c

let equal a b = compare a b = 0

let pp ppf d =
  match d.parameters_digest with
  | None -> Id.Codec.pp ppf d.codec
  | Some p ->
      Fmt.pf ppf "%a[%a]" Id.Codec.pp d.codec Digest.Dtype_parameters.pp p

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* codec = W.Obj.required o "codec" Codec.decode in
      let* parameters_digest =
        W.Obj.optional o "parameters_digest" Wire_digest.dtype_parameters
      in
      Ok { codec; parameters_digest })

let encode d =
  W.obj
    [
      W.field "codec" Codec.encode d.codec;
      W.field "parameters_digest"
        (W.enc_option Wire_digest.encode_dtype_parameters)
        d.parameters_digest;
    ]
