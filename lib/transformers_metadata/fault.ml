module Mismatch = struct
  type t = {
    actual : string;
    expected : string;
    field : string;
    identity : string;
  }
end

type error =
  [ `Constants_config_decode of string
  | `Metadata_duplicate of string
  | `Metadata_invalid of string
  | `Metadata_mismatch of Mismatch.t
  | `Metadata_missing of string
  | `Model_json_decode of string
  | `Source of string
  | `Weights_config_decode of string
  | Pt2_fixture_unix.Fault.error ]

let pp_error ppf : error -> unit = function
  | `Constants_config_decode s | `Model_json_decode s | `Weights_config_decode s
    ->
      Fmt.string ppf s
  | `Metadata_duplicate s -> Fmt.pf ppf "duplicate metadata identity: %s" s
  | `Metadata_invalid s -> Fmt.pf ppf "invalid metadata: %s" s
  | `Metadata_mismatch m ->
      Fmt.pf ppf "%s: %s is %s, expected %s" m.identity m.field m.actual
        m.expected
  | `Metadata_missing s -> Fmt.pf ppf "missing metadata: %s" s
  | `Source s -> Fmt.string ppf s
  | #Pt2_fixture_unix.Fault.error as e -> Pt2_fixture_unix.Fault.pp_error ppf e
