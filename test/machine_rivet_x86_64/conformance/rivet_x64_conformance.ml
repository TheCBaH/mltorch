(* Per-form conformance for the admitted x86-64 forms: each form made as typed
   Rivet instructions, run under qemu-user (emulation, not an x86-64 CPU), and
   compared with the interpreter's semantics under each form's defined-bit,
   flag and write masks.

   [--mutate NAME] makes the model deliberately wrong in one entry; the run
   must then report mismatches. [--form SUBSTRING] runs the forms whose name
   contains it. *)

open Machine_ir
open X64_forms
open X64_model
module B = X64_batch

let hex = Printf.sprintf "0x%Lx"

type stats = {
  mutable forms : int;
  mutable vectors : int;
  mutable skipped : int;
  mutable mismatches : int;
  mutable refused : int;
}

let check_one (f : X64_forms.t) (v : vector) (p : prediction)
    ((tys : Mir_type.t list), (d : B.placed)) (o : B.observed) =
  let problems = ref [] in
  let bad fmt = Fmt.kstr (fun s -> problems := s :: !problems) fmt in
  let non_flag =
    List.filter (fun (t : Mir_type.t) -> t <> Mir_type.Flags) tys
  in
  ignore d;
  ignore non_flag;
  let j = ref 0 in
  let flags_seen = ref false in
  let compact = compact_flags o.B.rflags in
  List.iteri
    (fun idx (e : expected) ->
      match e with
      | Flags_out { bits; defined } ->
          flags_seen := true;
          if
            not
              (Int64.equal
                 (Int64.logand (Int64.logxor compact bits) defined)
                 0L)
          then
            bad "flags %s, expected %s under %s" (hex compact) (hex bits)
              (hex defined)
      | Flags_kept ->
          if not (Int64.equal compact v.flags) then
            bad "flags %s, expected them kept at %s" (hex compact) (hex v.flags)
      | Flags_unknown -> ()
      | Gpr x ->
          let lo, _ = List.nth o.B.outs !j in
          incr j;
          if not (Int64.equal lo x) then
            bad "result %d = %s, expected %s" idx (hex lo) (hex x)
      | Ptr_delta dl ->
          let lo, _ = List.nth o.B.outs !j in
          incr j;
          if not (Int64.equal (Int64.sub lo o.B.base) dl) then
            bad "pointer result - base = %Ld, expected %Ld"
              (Int64.sub lo o.B.base) dl
      | Xmm { ty; lo; hi; mask_lo; mask_hi } ->
          let olo, ohi = canonical_xmm ty (List.nth o.B.outs !j) in
          let lo, hi = canonical_xmm ty (lo, hi) in
          incr j;
          if
            not
              (Int64.equal (Int64.logand (Int64.logxor olo lo) mask_lo) 0L
              && Int64.equal (Int64.logand (Int64.logxor ohi hi) mask_hi) 0L)
          then
            bad "xmm result = %s:%s, expected %s:%s (mask %s:%s)" (hex ohi)
              (hex olo) (hex hi) (hex lo) (hex mask_hi) (hex mask_lo))
    p.results;
  ignore !flags_seen;
  if o.B.buffer <> p.buffer then bad "buffer differs";
  ignore f;
  List.rev !problems

let describe (v : vector) =
  String.concat " "
    (List.map
       (function
         | Some (o : operand) -> Printf.sprintf "%Lx:%Lx" o.hi o.lo
         | None -> "-")
       v.inputs)
  ^ Printf.sprintf " fl=%Lx idx=%Ld seed=%Lx:%Lx" v.flags v.index v.seed_hi
      v.seed_lo

let run_form ~mutation ~map_mutation ~gnu ~verbose st (f : X64_forms.t) =
  let vs = X64_model.vectors f in
  let usable =
    List.filter_map
      (fun v ->
        match X64_model.predict mutation f v with
        | p -> Some (v, p)
        | exception Model_defect _ -> None)
      vs
  in
  (* the instructions do not depend on how many vectors a batch holds *)
  let usable = if gnu then List.filteri (fun i _ -> i < 4) usable else usable in
  st.forms <- st.forms + 1;
  st.skipped <- st.skipped + (List.length vs - List.length usable);
  match B.build ?mutation:map_mutation f (List.map fst usable) with
  | Error e ->
      st.refused <- st.refused + 1;
      Fmt.pr "%-28s refused: %s@." f.name e
  | Ok built when gnu -> (
      match
        Machine_rivet_x86_64_gnu.Gnu_coherence.check ~entry:"_start"
          [ built.B.modul ]
      with
      | Machine_rivet_x86_64_gnu.Gnu_coherence.Verdict.Agree _ ->
          st.vectors <- st.vectors + built.B.count
      | v ->
          st.mismatches <- st.mismatches + 1;
          let text =
            Fmt.str "%a" Machine_rivet_x86_64_gnu.Gnu_coherence.Verdict.pp v
          in
          Fmt.pr "%-28s %s@." f.name
            (if String.length text > 300 then String.sub text 0 300 else text))
  | Ok built -> (
      match
        Result.bind (B.elf built) Machine_rivet_x86_64.Rivet_x64_qemu.execute
      with
      | Error e ->
          st.refused <- st.refused + 1;
          Fmt.pr "%-28s did not run: %s@." f.name e
      | Ok out ->
          if String.length out <> built.B.count * B.stride then begin
            st.refused <- st.refused + 1;
            Fmt.pr "%-28s wrote %d bytes, expected %d@." f.name
              (String.length out) (built.B.count * B.stride)
          end
          else begin
            let places = B.result_places f in
            let bads = ref 0 in
            List.iteri
              (fun r (v, p) ->
                let o = B.observe (snd places) out r in
                match check_one f v p places o with
                | [] -> ()
                | probs ->
                    incr bads;
                    if !bads <= 2 || verbose then
                      Fmt.pr "%-28s %s@.    %s@." f.name (describe v)
                        (String.concat "; " probs))
              usable;
            st.vectors <- st.vectors + List.length usable;
            st.mismatches <- st.mismatches + !bads;
            if !bads > 0 then
              Fmt.pr "%-28s %d of %d vectors differ@." f.name !bads
                (List.length usable)
          end)

let () =
  let mutate = ref None
  and map_mutate = ref None
  and only = ref None
  and verbose = ref false
  and gnu = ref false in
  Arg.parse
    [
      ( "--mutate",
        Arg.String (fun s -> mutate := Some s),
        "NAME make the model wrong in one entry" );
      ( "--map-mutate",
        Arg.String (fun s -> map_mutate := Some s),
        "NAME make the Rivet mapping wrong in one entry" );
      ( "--gnu",
        Arg.Set gnu,
        " check each form's module against GNU as/ld instead of running it" );
      ("--form", Arg.String (fun s -> only := Some s), "SUBSTRING");
      ("--verbose", Arg.Set verbose, " print every mismatch");
    ]
    (fun _ -> ())
    "rivet_x64_conformance";
  let mutation =
    Option.map
      (fun n ->
        match List.assoc_opt n X64_model.Mutation.all with
        | Some m -> m
        | None -> failwith ("unknown mutation " ^ n))
      !mutate
  in
  let map_mutation =
    Option.map
      (function
        | "dropped-disp" ->
            Machine_rivet_x86_64.Rivet_x64_form.Mutation.Dropped_disp
        | "inverted-cond" ->
            Machine_rivet_x86_64.Rivet_x64_form.Mutation.Inverted_cond
        | n -> failwith ("unknown mapping mutation " ^ n))
      !map_mutate
  in
  let contains s sub =
    let n = String.length sub in
    let rec go i =
      i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
    in
    go 0
  in
  let forms =
    List.filter
      (fun (f : X64_forms.t) ->
        match !only with None -> true | Some s -> contains f.name s)
      X64_forms.all
  in
  let st =
    { forms = 0; vectors = 0; skipped = 0; mismatches = 0; refused = 0 }
  in
  List.iter
    (run_form ~mutation ~map_mutation ~gnu:!gnu ~verbose:!verbose st)
    forms;
  Fmt.pr
    "%d forms, %d vectors (emulated under qemu-user), %d skipped by the \
     model's defects, %d mismatches, %d forms not run@."
    st.forms st.vectors st.skipped st.mismatches st.refused;
  let failed = st.mismatches > 0 || st.refused > 0 in
  match (mutation, map_mutation) with
  | None, None -> exit (if failed then 1 else 0)
  | _ -> exit (if failed then 0 else 1)
