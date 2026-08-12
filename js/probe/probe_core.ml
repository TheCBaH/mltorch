(* Section 2: the error framework. Reachable from melange (core depends on fmt
   alone).

   The backtrace needs care. [Err.fail] captures a callstack under the active
   [Err.Config] policy, and [Err.Stack.pp] has two branches: frames when the
   runtime supplies them (native), and "stack unavailable" when it does not.
   BOTH JS runtimes take the second branch -- js_of_ocaml drops the callstack,
   and melange's [backtrace_slots] returns [None]. So printing slot counts, or
   whether slots exist at all, would make this probe differ between backends BY
   DESIGN and the differential diff could never pass.

   Instead we print one backend-neutral verdict: whichever branch was taken, was
   it well formed? Natively that means real frames; on JS it means the fallback
   notice actually rendered. Either way the answer is [true], and a genuine
   regression -- an empty slot array, or a missing notice -- still flips it. *)

let backtrace_path_valid (e : string Err.Error.t) =
  let rendered = Core.Pretty.to_string (Err.Error.pp Fmt.string) e in
  let contains hay needle =
    let nh = String.length needle and n = String.length hay in
    let rec go i =
      i + nh <= n && (String.sub hay i nh = needle || go (i + 1))
    in
    go 0
  in
  let slots =
    match Err.Error.origin e with
    | None -> None
    | Some origin -> (
        match Err.Origin.stack origin with
        | None -> None
        | Some stack -> (
            match Err.Stack.to_raw_backtrace stack with
            | None -> None
            | Some bt -> Printexc.backtrace_slots bt))
  in
  match slots with
  | Some slots -> Array.length slots > 0
  | None ->
      contains rendered "stack unavailable"
      || contains rendered "not linked with -g"

let run () =
  print_endline "=== core ===";
  let failed : (int, string) Err.t = Err.fail "boom" in
  let bridged : (int, string) Err.t = Err.of_option "absent" None in
  let ok : (int, string) Err.t = Err.return 7 in
  List.iter
    (fun (label, r) ->
      Printf.printf "%s: %s\n" label
        (Core.Pretty.to_string
           (Core.Pretty.err_result ~ok:Fmt.int ~error:Fmt.string)
           r))
    [ ("fail", failed); ("of_option", bridged); ("return", ok) ];
  (* Print the verdict, then RAISE on it. Both backends run this source, so a
     [false] here -- or an [Ok] from a constructor that must fail -- reproduces
     identically, diffs clean and exits 0. Printing a verdict is not checking
     it; see [[testing_strategy]]. *)
  List.iter
    (fun (label, r) ->
      match r with
      | Ok _ ->
          failwith
            (Printf.sprintf "probe: %s returned Ok; it must produce an error"
               label)
      | Error e ->
          let valid = backtrace_path_valid e in
          Printf.printf "%s backtrace-path-valid=%b\n" label valid;
          if not valid then
            failwith
              (Printf.sprintf
                 "probe: %s took neither backtrace branch cleanly (slots \
                  present but empty, or the unavailable notice missing)"
                 label))
    [ ("fail", failed); ("of_option", bridged) ]
