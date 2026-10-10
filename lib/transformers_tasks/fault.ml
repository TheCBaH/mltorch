type error =
  [ Transformers_metadata.Fault.error | Pt2_fixture_unix.Fixture.error ]

let pp ppf : error -> unit = function
  | #Transformers_metadata.Fault.error as e ->
      Transformers_metadata.Fault.pp_error ppf e
  | #Pt2_fixture_unix.Fixture.error as e ->
      Pt2_fixture_unix.Fixture.pp_error ppf e
