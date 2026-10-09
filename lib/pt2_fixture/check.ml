(* The comparisons every layer repeats: a scalar, a pinned file, a digest, a
   size. Each names the layer it concerns. *)

open Err.Syntax
module Pin = Pt2_checkpoint_map.Document.Pin

let field layer field ~actual ~expected =
  if String.equal actual expected then Err.return ()
  else
    Err.fail (`Field_clash { Fault.Field_clash.layer; field; actual; expected })

let digest layer actual expected =
  if Pt2_sha256.Digest.equal actual expected then Err.return ()
  else Err.fail (`Digest_clash { Fault.Digest_clash.layer; actual; expected })

let size layer actual expected =
  if Int64.equal actual expected then Err.return ()
  else Err.fail (`Size_clash { Fault.Size_clash.layer; actual; expected })

let pin layer (actual : Pin.t) (expected : Pin.t) =
  let* () = digest layer actual.sha256 expected.sha256 in
  let* () = size layer actual.size expected.size in
  let* () =
    field layer Fault.File_name ~actual:actual.name ~expected:expected.name
  in
  field layer Fault.Url ~actual:actual.url ~expected:expected.url
