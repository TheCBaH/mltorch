module W = Wire

type t = Finite_value of float | No_finite_value | Overflow

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "finite" ->
          let* v = W.Obj.required o "value" W.float in
          Ok (Finite_value v)
      | "no_finite_value" -> Ok No_finite_value
      | "overflow" -> Ok Overflow
      | tag -> W.unknown_tag ~kind:"error_metric" tag)

let encode = function
  | Finite_value v -> W.tagged "finite" [ W.field "value" W.enc_float v ]
  | No_finite_value -> W.tagged "no_finite_value" []
  | Overflow -> W.tagged "overflow" []
