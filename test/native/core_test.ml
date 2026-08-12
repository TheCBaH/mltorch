(* Exercises the lib/core error framework against lib/native's local error sets:
   deterministic payload messages (golden), composition by row-union + printer
   delegation, automatic backtrace capture (structural, never golden — line
   numbers shift), and let* short-circuit + row unification. *)

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

(* These tests run natively AND under js_of_ocaml (see the [modes] field in
   test/native/dune), and the two backends legitimately disagree about
   backtraces: native captures real frames, while js_of_ocaml drops the
   callstack so no usable stack survives and [Err.Stack.pp] renders its
   "stack unavailable" notice instead. Both are correct, and one [%expect]
   golden cannot hold two spellings.

   So assert that whichever branch was taken is WELL FORMED, rather than
   asserting one backend's branch. Natively that still means real frames; on JS
   it means the fallback notice actually rendered. A genuine regression -- an
   empty slot array, or a missing notice -- still flips these to false. Same
   technique as js/probe/probe_core.ml. *)

(* The detection stack, as slots, when this backend and build produced usable
   ones. Three things can each legitimately yield [None]: the policy captured no
   origin ([Err.Config.fast]), the origin holds no stack, or the stack has no
   symbols. [Err.Stack.is_available] answers only the second, so go through
   [to_raw_backtrace] and let [Printexc] answer the third -- a native binary
   built without -g captures frames it cannot render, and that must take the
   same branch as JS rather than assert on a notice. *)
let frames e =
  match Err.Error.origin e with
  | None -> None
  | Some origin -> (
      match Err.Origin.stack origin with
      | None -> None
      | Some stack -> (
          match Err.Stack.to_raw_backtrace stack with
          | None -> None
          | Some bt -> Printexc.backtrace_slots bt))

(* The two spellings of "there is nothing to show here": [Err]'s own notice for
   an empty stack, and the runtime's for a stack it cannot symbolize. *)
let no_frames_notice rendered =
  contains rendered "stack unavailable"
  || contains rendered "not linked with -g"

let backtrace_path_valid e rendered =
  match frames e with
  | Some slots -> Array.length slots > 0
  | None -> no_frames_notice rendered

(* Whether the calling file is named among the frames. Where there are no frames
   at all the claim is unobservable rather than false, so the neutral form is:
   when slots exist the caller must be named, otherwise the notice must be. *)
let caller_named e rendered file =
  match frames e with
  | Some _ -> contains rendered file
  | None -> no_frames_notice rendered

let pp_dim_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:Dim.pp_error

(* Deterministic payload messages — the real regression guard. *)
let%expect_test "component pp_error messages" =
  Format.printf "%a@." Dim.pp_error (`Non_positive_extent (-3));
  Format.printf "%a@." Aten_shape.pp_error
    (`Rank_out_of_range { Aten_shape.rank = 7; lo = 0; hi = 6 });
  Format.printf "%a@." Symint.pp_error
    (`Out_of_range { Symint.name = "s0"; value = 17; lo = 1; hi = 1024 });
  Format.printf "%a@." Symint.pp_error
    (`Not_divisible { Symint.name = "s0"; value = 17; divisor = 8 });
  [%expect
    {|
    extent must be >= 1, got -3
    rank 7 out of [0, 6]
    symint s0 = 17 out of [1, 1024)
    symint s0 = 17 not divisible by 8 |}]

(* Extents are sizes, valid from 1: zero is rejected too, not only negatives —
   the engine has no empty tensors. *)
let%expect_test "extent floor is 1 (zero rejected)" =
  (match Dim.extent 0 with
  | _ -> print_string "no error\n"
  | exception Invalid_argument msg -> Printf.printf "raised: %s\n" msg);
  Format.printf "%a"
    (Core.Pretty.err_result ~ok:(Fmt.any "ok\n") ~error:(fun ppf e ->
         Fmt.pf ppf "%a@." Dim.pp_error e))
    (Dim.extent_checked 0);
  [%expect
    {|
    raised: Dim.extent: must be >= 1
    extent must be >= 1, got 0 |}]

(* Composition: Aten_shape.error unions Dim.error, and its printer delegates the
   [#Dim.error] tag back to Dim.pp_error. Both values are typed [Aten_shape.error]. *)
let%expect_test "Aten_shape.pp_error delegates Dim tags to Dim.pp_error" =
  let own : Aten_shape.error =
    `Rank_out_of_range { Aten_shape.rank = 9; lo = 0; hi = 6 }
  in
  let from_dim : Aten_shape.error = `Non_positive_extent (-2) in
  Format.printf "%a@." Aten_shape.pp_error own;
  Format.printf "%a@." Aten_shape.pp_error from_dim;
  [%expect {|
    rank 9 out of [0, 6]
    extent must be >= 1, got -2 |}]

(* [fail] captures a stack automatically; [Err.Error.pp] renders message +
   provenance. Assert structure, not backtrace content (it shifts per build). *)
let%expect_test "fail captures backtrace; pp renders message + 'detected at:'" =
  match Dim.extent_checked (-3) with
  | Ok _ -> Format.printf "unexpected Ok@."
  | Error e ->
      let s = Core.Pretty.to_string (Err.Error.pp Dim.pp_error) e in
      assert (backtrace_path_valid e s);
      Format.printf "msg=%b provenance=%b@."
        (contains s "extent must be >= 1, got -3")
        (contains s "detected at:");
      [%expect {| msg=true provenance=true |}]

(* [of_option] is the option-shaped [fail]: the point of it over
   [Stdlib.Option.to_result] is that the failure carries a [Err.Error.t], so a
   bridged absence has provenance like any other. *)
let%expect_test "of_option bridges an option into the framework" =
  let pp ppf (`Missing k) = Format.fprintf ppf "no binding for %s" k in
  let show o =
    Format.printf "%a@."
      (Core.Pretty.err_result ~ok:Format.pp_print_int ~error:pp)
      (Err.of_option (`Missing "k") o)
  in
  show (Some 7);
  show None;
  [%expect {|
    7
    no binding for k |}]

(* The reason [of_option] is a [Core] helper and not [Option.to_result]: the
   CALLER must stay visible in the backtrace. A version that built the [Error.t]
   by hand with an empty backtrace would satisfy every assertion above and lose
   exactly this.

   Asserts that the calling file is named, not a frame index. [of_option] adds
   its own frame between [make] and the caller, and inlining decides which of
   those survives — on 4.14.3 [fail] inlines into [of_option] while [of_option]
   itself does not. Naming an index here would encode one compiler's choices. *)
let%expect_test "of_option keeps the caller in the backtrace" =
  let pp ppf (`Missing k) = Format.fprintf ppf "no binding for %s" k in
  match Err.of_option (`Missing "k") None with
  | Ok () -> Format.printf "unexpected Ok@."
  | Error e ->
      let s = Core.Pretty.to_string (Err.Error.pp pp) e in
      Format.printf "captured=%b caller_named=%b@." (backtrace_path_valid e s)
        (caller_named e s "core_test.ml");
      [%expect {| captured=true caller_named=true |}]

(* let* stops at the first Error; the row unifies to the union of every tag the
   chain can raise (it only type-checks because Aten_shape.error holds both). *)
let%expect_test "let* short-circuits and unifies the error row" =
  let open Err.Syntax in
  let chain ~rank_bad : (int, Aten_shape.error) Err.t =
    let* () =
      if rank_bad then
        Err.fail (`Rank_out_of_range { Aten_shape.rank = 7; lo = 0; hi = 6 })
      else Err.return ()
    in
    let* () = Err.fail (`Non_positive_extent (-1)) in
    Err.return 99 (* unreached *)
  in
  let show ~rank_bad =
    Format.printf "%a"
      (Core.Pretty.err_result
         ~ok:(fun ppf n -> Fmt.pf ppf "ok %d@." n)
         ~error:(fun ppf e -> Fmt.pf ppf "%a@." Aten_shape.pp_error e))
      (chain ~rank_bad)
  in
  show ~rank_bad:true;
  show ~rank_bad:false;
  [%expect {|
    rank 7 out of [0, 6]
    extent must be >= 1, got -1 |}]

(* The remaining one-line combinators: >>|, >>=, map_error (lifts the payload,
   keeps the backtrace), and Err.List.map (short-circuits on first Error). *)
let%expect_test "combinators: >>| >>= map_error Err.List.map" =
  let open Err.Syntax in
  let r1 = (Err.return 4 >>| fun x -> x + 1) >>= fun x -> Err.return (x * 2) in
  let lifted =
    Err.fail (`Non_positive_extent (-5)) |> Err.map_error (fun _ -> `Renamed)
  in
  let pos x = if x >= 0 then Err.return x else Err.fail (`Neg x) in
  let summary =
    let* r1n = r1 in
    let lifted_name =
      match lifted with
      | Ok () -> "ok"
      | Error e -> if Err.Error.kind e = `Renamed then "Renamed" else "?"
    in
    Err.return
      ( r1n,
        lifted_name,
        Err.List.map pos [ 1; 2; 3 ] = Ok [ 1; 2; 3 ],
        match Err.List.map pos [ 1; -2; 3 ] with
        | Error _ -> true
        | Ok _ -> false )
  in
  let pp_summary ppf (r1n, lifted, listok, listerr) =
    Format.fprintf ppf "r1=%d lifted=%s listok=%b listerr=%b" r1n lifted listok
      listerr
  in
  Format.printf "%a@." (pp_dim_result pp_summary) summary;
  [%expect {| r1=10 lifted=Renamed listok=true listerr=true |}]

(* --- traversal is stack-safe, and still left-to-right ---

   [Err.List] was a plain non-tail-recursive fold before it became [Err.List];
   [map] in particular rebuilt the result list on the way back up, so a long
   enough input raised [Stack_overflow] instead of returning. Nothing in the
   repo traverses 100k elements today, but a graph big enough to is exactly the
   input nobody tests with, and the failure mode is a crash rather than a wrong
   answer.

   The size is measured, not guessed. With the old [map] and an 8MB stack the
   overflow threshold sits between 200k and 300k elements, so 100_000 -- the
   obvious round number -- passes either way and proves nothing. 500_000 is
   past it with margin on both backends.

   Verified this can fail: restoring the old non-tail-recursive [map] turns
   this into "overflowed or failed". *)
let%expect_test "Err.List.map is stack-safe on a long list" =
  let n = 500_000 in
  let input = List.init n Fun.id in
  let doubled =
    match Err.List.map (fun x -> Err.return (x * 2)) input with
    | Ok xs -> Some xs
    | Error (_ : _ Err.Error.t) -> None
    | exception Stack_overflow -> None
  in
  (match doubled with
  | None -> print_endline "overflowed or failed"
  | Some xs ->
      Printf.printf "len=%d first=%d last=%d\n" (List.length xs) (List.hd xs)
        (List.nth xs (n - 1)));
  [%expect {| len=500000 first=0 last=999998 |}]

(* Order is a separate claim from safety: a tail-recursive [map] that folded the
   input and reversed the accumulator would pass the test above while running
   the callback right-to-left, which matters the moment the callback has an
   effect -- and the first error reported for a list with several would become
   the LAST one. Records the call order and the short-circuit point together, so
   one golden pins both. *)
let%expect_test
    "Err.List traversal runs left to right and stops at the first error" =
  let seen = ref [] in
  let note x = seen := x :: !seen in
  let r =
    Err.List.map
      (fun x ->
        note x;
        if x = 3 then Err.fail (`Stopped x) else Err.return x)
      [ 1; 2; 3; 4; 5 ]
  in
  let stopped_at =
    match r with
    | Ok _ -> "none"
    | Error e -> ( match Err.Error.kind e with `Stopped x -> string_of_int x)
  in
  Printf.printf "visited=[%s] stopped_at=%s\n"
    (String.concat ";" (List.rev_map string_of_int !seen))
    stopped_at;
  (* [iter] and [fold_left] short-circuit at the same element. *)
  let seen_iter = ref 0 in
  let ri =
    Err.List.iter
      (fun x ->
        incr seen_iter;
        if x = 3 then Err.fail (`Stopped x) else Err.return ())
      [ 1; 2; 3; 4; 5 ]
  in
  let seen_fold = ref 0 in
  let rf =
    Err.List.fold_left
      (fun acc x ->
        incr seen_fold;
        if x = 3 then Err.fail (`Stopped x) else Err.return (acc + x))
      0 [ 1; 2; 3; 4; 5 ]
  in
  Printf.printf
    "iter_visited=%d iter_failed=%b fold_visited=%d fold_failed=%b\n" !seen_iter
    (Result.is_error ri) !seen_fold (Result.is_error rf);
  [%expect
    {|
    visited=[1;2;3] stopped_at=3
    iter_visited=3 iter_failed=true fold_visited=3 fold_failed=true |}]

(* --- explicit positions ---

   [~pos:__POS__] is the provenance that survives where a callstack does not.
   Under js_of_ocaml -- which this suite also runs under -- no stack is
   captured at all, so a mapping recorded with only a stack policy names
   nothing; a position is compiled in and reads the same on every backend.
   That is why the subsystem seams (Pt2_archive, Native_interp, Me_export)
   carry one.

   Pinned under an explicit policy rather than the ambient one. [Err.Config] is
   process-wide, so a test that changed it and left it changed would leak into
   every later expect test in this executable -- hence [Fun.protect]. *)
let with_config config f =
  let saved = Err.Config.get () in
  Err.Config.set config;
  Fun.protect ~finally:(fun () -> Err.Config.set saved) f

let%expect_test "a mapped position is recorded on every backend" =
  (* [backtrace = Never] is the honest model of the browser: whatever the
     policy asks for, no frames exist there. If [~pos] were not recorded, this
     error would render with no provenance whatsoever. *)
  let no_stacks =
    match
      Err.Config.make ~actions:Err.Action.Set.boundaries
        ~backtrace:Err.Config.Never ~max_events:8 ~max_frames:0
        ~max_external_bytes:1024
    with
    | Ok c -> c
    | Error e ->
        failwith
          (Core.Pretty.to_string Err.Config.pp_make_error (Err.Error.kind e))
  in
  with_config no_stacks (fun () ->
      let pp ppf (`Wrapped (`Inner k)) = Format.fprintf ppf "wrapped %s" k in
      let mapped =
        Err.map_error ~pos:__POS__
          (fun inner -> `Wrapped inner)
          (Err.fail (`Inner "boom"))
      in
      match mapped with
      | Ok () -> print_endline "unexpected Ok"
      | Error e ->
          let s = Core.Pretty.to_string (Err.Error.pp pp) e in
          Printf.printf "payload=%b no_origin=%b mapped=%b names_file=%b\n"
            (contains s "wrapped boom")
            (not (contains s "detected at"))
            (contains s "mapped")
            (contains s "core_test.ml"));
  [%expect {| payload=true no_origin=true mapped=true names_file=true |}]

(* The policy really is restored: the previous test would otherwise leave every
   later error in this executable stackless. *)
let%expect_test "with_config restores the ambient policy" =
  let before = Err.Config.backtrace (Err.Config.get ()) in
  (try with_config Err.Config.debug (fun () -> failwith "boom")
   with Failure _ -> ());
  let after = Err.Config.backtrace (Err.Config.get ()) in
  Printf.printf "restored=%b\n" (before = after);
  [%expect {| restored=true |}]
