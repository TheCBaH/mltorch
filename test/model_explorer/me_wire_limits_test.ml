(* [Me_limits.Wire_limits.jsont]: the profile that crosses the worker boundary
   (.ai/model_explorer_design.md).

   The request IS the wire, so "the worker and the page disagree about the
   profile" has to be unconstructable rather than merely rejected somewhere.
   Three properties carry that, and none is visible in a round trip that
   happens to succeed: only tightenings decode, an unknown member is refused by
   name rather than ignored, and every value is bounded BEFORE it is narrowed to
   its field's width. *)

(* Jsont styles its decode errors with ANSI escapes unless the ambient
   environment says otherwise, and cram/expect goldens below quote that text.
   Pinned here so a bare [dune runtest] agrees with [make runtest], which sets
   NO_COLOR itself -- a golden that depends on the caller's environment is a
   golden that passes for the wrong reason. *)
let () = Jsont.Error.disable_ansi_styler ()

module L = Me_limits.Limits
module W = Me_limits.Wire_limits

let decode = Jsont_bytesrw.decode_string W.jsont
let encode = Jsont_bytesrw.encode_string W.jsont

(* Jsont hands back a [(_, string) result], outside the framework. Named, per
   CLAUDE.md, so it reads as a deliberate crossing rather than a dropped
   wrapper. *)
let pp_wire ppf r =
  Core.Pretty.result ~ok:Fmt.string
    ~error:(fun ppf e -> Fmt.pf ppf "REJECTED %s" e)
    ppf r

let show_json t = Format.printf "%a@." pp_wire (encode t)

let wire l =
  Err.or_raise ~pp_error:Me_limits.pp_error (W.of_limits ~ceiling:L.untrusted l)

let tighten f = wire (Err.or_raise ~pp_error:Me_limits.pp_error (f L.untrusted))

(* --- what crosses --- *)

let%expect_test "the ceiling itself carries no fields" =
  (* A wire profile is by construction no looser than [untrusted], so a field
     equal to it says nothing. Encoding only the differences is what makes two
     equal profiles encode to equal bytes. *)
  show_json (wire L.untrusted);
  [%expect {| {} |}]

let%expect_test "a tightening carries exactly what it tightens" =
  show_json (tighten (L.create ~max_nodes_per_graph:64 ~max_views:2));
  [%expect {| {"max_nodes_per_graph":"64","max_views":"2"} |}]

let%expect_test "a nested archive field is flat and dotted" =
  (* [zip.max_entries] is the name [Me_limits.Invalid.t] already uses, so the
     member a rejection names and the member that carried it are the same
     string rather than two tables that agree today. *)
  let z =
    Err.or_raise ~pp_error:Pt2_zip.Limits.pp_error
      (Pt2_zip.Limits.create ~max_entries:8 ~max_entry_bytes:0x1000L
         ~max_total_bytes:0x1_0000L ~max_path_bytes:64 ~max_path_depth:4)
  in
  show_json (tighten (L.create ~zip:z));
  [%expect
    {| {"zip.max_entries":"8","zip.max_entry_bytes":"4096","zip.max_path_bytes":"64","zip.max_path_depth":"4","zip.max_total_bytes":"65536"} |}]

(* Encode, decode, encode again. Equal bytes on both sides is the round trip
   AND the determinism claim in one statement -- the encoding is a function of
   the profile, so two spellings of one profile cannot survive it. *)
let round_trip t =
  Result.bind (encode t) (fun text ->
      Result.bind (decode text) (fun back ->
          Result.map
            (fun again -> (text, String.equal text again))
            (encode back)))

let%expect_test "small round-trips through its own encoding" =
  (* Counted rather than printed: [small] tightens most of the table, and a
     golden holding all 36 members would be re-promoted without being read on
     every recalibration. The SHAPE is pinned by the two tests above; what this
     adds is that the whole profile survives the trip unchanged. Splitting on
     ',' is exact here because every value is a quoted decimal. *)
  Format.printf "%a@."
    (Core.Pretty.result
       ~ok:(fun ppf (text, same) ->
         Fmt.pf ppf "%d members, stable: %b"
           (List.length (String.split_on_char ',' text))
           same)
       ~error:Fmt.string)
    (round_trip (wire L.small));
  [%expect {| 36 members, stable: true |}]

(* --- what does not cross --- *)

let%expect_test "a value the ceiling refuses" =
  (* [untrusted] is the ceiling, not [Hard]: a caller may tighten it and never
     weaken it, and a value below [Hard] can be far looser than [untrusted]. *)
  Format.printf "%a@." pp_wire
    (Result.map
       (fun _ -> "accepted")
       (decode {|{"max_nodes_per_graph":"1048576"}|}));
  [%expect {| REJECTED invalid limit max_nodes_per_graph = 1048576 |}]

let%expect_test "a value that would WRAP before any check saw it" =
  (* 2^32 + 1 narrows to 1 under js_of_ocaml's 32-bit [int]. Bounding the
     [int64] first is what makes this a rejection on both backends instead of a
     silently accepted profile of one node per graph on one of them -- and it is
     the reason [assign] is not callable outside [assign_checked].

     RED OBSERVED, and only on one backend: with [assign_checked]'s bound
     removed, [dune runtest] stays green -- natively the value does not wrap, so
     [create]'s own hard check still rejects it with the same message -- while
     [dune build @runtest-js] reports "accepted". That is exactly the split
     CLAUDE.md names: the backend diff sees divergence, the node expect run sees
     a wrong answer, and here only the second one can. *)
  Format.printf "%a@." pp_wire
    (Result.map
       (fun _ -> "accepted")
       (decode {|{"max_nodes_per_graph":"4294967297"}|}));
  [%expect {| REJECTED invalid limit max_nodes_per_graph = 4294967297 |}]

let%expect_test "an unknown member is refused, not ignored" =
  (* A profile the worker silently did not apply is exactly the disagreement
     this type exists to prevent, so an unrecognised key cannot be dropped. *)
  Format.printf "%a@." pp_wire
    (Result.map (fun _ -> "accepted") (decode {|{"max_nodes":"64"}|}));
  [%expect {| REJECTED invalid limit max_nodes = 64 |}]

let%expect_test "zero and negative are refused like any other out-of-range" =
  List.iter
    (fun v ->
      Format.printf "%-4s %a@." v pp_wire
        (Result.map
           (fun _ -> "accepted")
           (decode (Printf.sprintf {|{"max_views":"%s"}|} v))))
    [ "0"; "-1" ];
  [%expect
    {|
    0    REJECTED invalid limit max_views = 0
    -1   REJECTED invalid limit max_views = -1 |}]

let%expect_test "a JSON number is not a value here" =
  (* [int64_as_string] on purpose: two fields hold values past 2^31, and a JSON
     number would come back through a 32-bit [int] under node. *)
  Format.printf "%a@." pp_wire
    (Result.map (fun _ -> "accepted") (decode {|{"max_views":2}|}));
  [%expect
    {|
    REJECTED Expected int64 string but found number
    File "-", line 1, characters 13-14:
    File "-": in member max_views of
    File "-", line 1, characters 0-14: limits object |}]
