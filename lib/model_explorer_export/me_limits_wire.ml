(* The wire subset of profiles: [Wire_limits.t], checked against a ceiling
   and encoded/decoded as a flat tightening of [Limits.untrusted]. Split
   from me_limits.ml. Depends on me_limits_error.ml (for
   [pp_error]) and me_limits_profile.ml (for [Limits]). *)

open Me_limits_error
open Me_limits_profile

(* [Jsont.Object.as_string_map] yields a [Stdlib.Map.Make(String)], and
   [Map.Make] is applicative, so this local instance IS that type. *)
module Wire_map = Map.Make (String)

module Wire_limits = struct
  type t = Limits.t

  let of_limits ~ceiling l =
    let open Err.Syntax in
    let* () =
      Limits.check_against ~zip_ceiling:ceiling.Limits.zip
        ~field_ceiling:(fun f -> f.Limits.get ceiling)
        l
    in
    (* Through [Limits.create] as well, so that every wire profile is a
       create-validated profile whatever route produced [l]. *)
    Limits.create l

  let limits t = t

  (* The wire carries a TIGHTENING of [untrusted], never a whole profile: a
     [Wire_limits.t] is by construction no looser, so a field equal to
     [untrusted]'s carries no information and is not sent. [untrusted] itself
     therefore encodes as [{}].

     FLAT, keyed by the field's own diagnostic name -- [zip.max_entries], not a
     nested object -- so the member a rejection names and the member that
     carried it are literally the same string.

     Values are JSON STRINGS through [int64_as_string]: two of these fields are
     [int64] with values past 2^31, and a JSON number would be read back
     through a 32-bit [int] under jsoo. *)
  let member_jsont =
    Jsont.Object.as_string_map ~kind:"limits" Jsont.int64_as_string

  let jsont =
    Jsont.map ~kind:"wireLimits"
      ~dec:(fun members ->
        let pairs = Wire_map.bindings members in
        match
          (let open Err.Syntax in
           let* l =
             Limits.apply_overrides ~ceiling:Limits.untrusted
               ~base:Limits.untrusted pairs
           in
           (* [of_limits] rather than a bare [create]: the decoded value has to
              be a value the ENCODER could have produced, and only the ceiling
              check makes that true. *)
           of_limits ~ceiling:Limits.untrusted l)
          |> Err.export ~pos:__POS__
        with
        | Ok l -> l
        | Error k -> Jsont.Error.msgf Jsont.Meta.none "%a" pp_error k)
      ~enc:(fun t ->
        List.fold_left
          (fun m (name, v) -> Wire_map.add name v m)
          Wire_map.empty
          (Limits.overrides ~base:Limits.untrusted t))
      member_jsont
end
