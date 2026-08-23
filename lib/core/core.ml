(* See core.mli. *)

module Float_bits = Float_bits

module Pretty = struct
  let to_string pp v = Fmt.str "%a" pp v
  let pp_none none ppf () = Fmt.string ppf none
  let option_or ~none pp = Fmt.option ~none:(pp_none none) pp
  let result ~ok ~error = Fmt.result ~ok ~error
  let error_kind pp ppf e = pp ppf (Err.Error.kind e)
  let err_result ~ok ~error = Fmt.result ~ok ~error:(error_kind error)

  let capture_to_string ?like f =
    let buf = Buffer.create 256 in
    let ppf = Fmt.with_buffer ?like buf in
    f ppf;
    Fmt.flush ppf ();
    Buffer.contents buf
end
