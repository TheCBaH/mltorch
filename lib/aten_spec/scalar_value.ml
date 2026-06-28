(* A scalar literal: an op's Scalar argument, or one explicit tensor value.

   Canonical form is a single-key object so the int/float/bool tag is explicit:
   {"int": 1} | {"float": 1.0} | {"float": "0x80000000"} | {"bool": true}.
   As a convenience a bare JSON bool decodes to [Bool] and a bare number to
   [Float] (rounded to f32) — but a bare number can't be an [Int] (JSON has one
   number type), so use the object form when an integer scalar is required. *)

type t = Int of int | Float of Float32.t | Bool of bool

let jsont : t Jsont.t =
  Jsont.map ~kind:"scalar"
    ~dec:(fun json ->
      match json with
      | Jsont.Bool (b, _) -> Bool b
      | Jsont.Number (n, _) -> Float (Float32.to_f32 n)
      | Jsont.Object _ ->
          Spec_util.union ~kind:"scalar"
            [
              ("int", fun v -> Int (Spec_util.dec Jsont.int v));
              ("float", fun v -> Float (Spec_util.dec Float32.jsont v));
              ("bool", fun v -> Bool (Spec_util.dec Jsont.bool v));
            ]
            json
      | _ ->
          Jsont.Error.msgf Jsont.Meta.none
            "scalar: expected a number, bool, or single-key object")
    ~enc:(function
      | Int i -> Spec_util.single ~case:"int" (Spec_util.jint i)
      | Float f ->
          Spec_util.single ~case:"float" (Spec_util.enc Float32.jsont f)
      | Bool b -> Spec_util.single ~case:"bool" (Spec_util.jbool b))
    Jsont.json
