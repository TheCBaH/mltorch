(* Native per-form conformance for AArch64 (C3n). Every admitted form
   instance runs on this CPU inside a generated inline-assembly harness: the
   destination is seeded with a pattern so the write mask is visible, NZCV is
   seeded before and read right after the instruction, memory forms use a
   canaried buffer, and FPCR is fixed to round-to-nearest-even, no flush to
   zero, no default NaN and no traps for the run, then restored. Each result is
   compared with [A64_sem.exec] under the form's masks: exact integer bits,
   exact zeroed upper bits, exact NZCV, and, for a float result, any NaN for a
   NaN (the model does not claim payloads).

   [--mutate NAME] replaces one semantic entry with a deliberately wrong one;
   the run then succeeds only if the comparison catches it. A host that is not
   AArch64, or lacks FP/ASIMD, is reported as unavailable (exit 2), never as a
   pass. *)

open A64_forms
open A64_model

(* ---- driving ----------------------------------------------------------------- *)

let run_cmd cmd = Sys.command cmd

let () =
  let mutation =
    match Array.to_list Sys.argv with
    | [ _; "--mutate"; m ] -> (
        match List.assoc_opt m Mutation.all with
        | Some m -> Some m
        | None ->
            prerr_endline ("unknown mutation " ^ m);
            exit 64)
    | [ _ ] -> None
    | _ ->
        prerr_endline "usage: a64_conformance [--mutate NAME]";
        exit 64
  in
  let arch =
    String.trim (In_channel.input_all (Unix.open_process_in "uname -m"))
  in
  if arch <> "aarch64" then (
    Printf.printf "unavailable: host %s is not aarch64\n" arch;
    exit 2);
  let dir =
    let f = Filename.temp_file "a64_conformance" "" in
    Sys.remove f;
    Sys.mkdir f 0o700;
    f
  in
  let c = Filename.concat dir "harness.c"
  and exe = Filename.concat dir "harness" in
  Out_channel.with_open_text c (fun oc -> output_string oc (A64_c.c_program ()));
  if
    run_cmd
      (Printf.sprintf "gcc -O1 -o %s %s" (Filename.quote exe) (Filename.quote c))
    <> 0
  then (
    prerr_endline "the harness does not compile";
    exit 1);
  let vectors = A64_model.vectors () in
  let seed = A64_model.seed and random_per_form = A64_model.random_per_form in
  let input = Filename.concat dir "in.txt"
  and output = Filename.concat dir "out.txt" in
  Out_channel.with_open_text input (fun oc ->
      List.iter
        (fun (k, _, ins, nz, seed, buf) ->
          Printf.fprintf oc "%d %Lx %Lx %Lx %Lx %Lx %Lx %Lx %Lx %Lx\n" k
            (List.nth ins 0) (List.nth ins 1) (List.nth ins 2) nz seed
            (Bytes.get_int64_le buf 0) (Bytes.get_int64_le buf 8)
            (Bytes.get_int64_le buf 16)
            (Bytes.get_int64_le buf 24))
        vectors);
  let status =
    run_cmd
      (Printf.sprintf "%s < %s > %s" (Filename.quote exe) (Filename.quote input)
         (Filename.quote output))
  in
  if status = 2 then (
    print_endline "unavailable: the host lacks FP/ASIMD";
    exit 2);
  if status <> 0 then (
    prerr_endline "the harness failed";
    exit 1);
  let lines =
    In_channel.with_open_text output (fun ic ->
        String.split_on_char '\n' (In_channel.input_all ic)
        |> List.filter (( <> ) ""))
  in
  let fpcr, results = match lines with f :: r -> (f, r) | [] -> ("?", []) in
  let addresses, results =
    List.partition
      (fun l -> String.length l > 5 && String.sub l 0 5 = "addr ")
      results
  in
  (* the model: a region at a page-aligned base, the view at the target's
     offset; ADRP gives the region offset with its low twelve bits cleared and
     ADD :lo12: adds them back *)
  let address_failures =
    List.filter_map
      (fun l ->
        match String.split_on_char ' ' l with
        | [ _; k; target; page; full ] ->
            let h x = Int64.of_string ("0x" ^ x) in
            let target = h target and page = h page and full = h full in
            let base = Int64.logand target (Int64.lognot 0xFFFL) in
            let offset = Int64.sub target base in
            let model_page =
              Int64.add base (Int64.logand offset (Int64.lognot 0xFFFL))
            in
            let model_full =
              Int64.add model_page (Int64.logand offset 0xFFFL)
            in
            if Int64.equal page model_page && Int64.equal full model_full then
              None
            else
              Some
                (Printf.sprintf
                   "adrp/add_lo12 +%s: native %Lx %Lx, model %Lx %Lx" k page
                   full model_page model_full)
        | _ -> Some ("unreadable " ^ l))
      addresses
  in
  let gcc =
    String.trim
      (In_channel.input_all (Unix.open_process_in "gcc --version | head -1"))
  in
  Printf.printf
    "aarch64 per-form conformance: %d forms, %d vectors (all boundary pairs, \
     %d random each), seed %d\n"
    (List.length forms) (List.length vectors) random_per_form seed;
  Printf.printf "host: %s; %s; %s\n" arch gcc fpcr;
  let failures = Hashtbl.create 16 in
  List.iter2
    (fun (_, (f : Form.t), ins, nz, seed, buf) line ->
      match
        String.split_on_char ' ' line
        |> List.map (fun s -> Int64.of_string ("0x" ^ s))
      with
      | [ lo; hi; flags; m0; m1; m2; m3 ] -> (
          let native_mem = Bytes.create 32 in
          List.iteri
            (fun i w -> Bytes.set_int64_le native_mem (8 * i) w)
            [ m0; m1; m2; m3 ];
          match predict mutation f ins nz seed buf with
          | exception Model_defect _ ->
              Hashtbl.replace failures f.Form.name
                (Printf.sprintf "model defect at inputs %s"
                   (String.concat "," (List.map (Printf.sprintf "%Lx") ins)))
          | plo, phi, pflags, pmem ->
              let result_ok =
                match f.Form.result with
                | Some (F _ as r)
                  when float_bits_nan r lo && float_bits_nan r plo ->
                    Int64.equal hi phi
                | _ -> Int64.equal lo plo && Int64.equal hi phi
              in
              let mem_ok =
                Array.for_all Fun.id
                  (Array.init 32 (fun i ->
                       Char.code (Bytes.get native_mem i) = pmem.(i)))
              in
              let flags_ok =
                Int64.equal
                  (Int64.logand flags 0xF000_0000L)
                  (Int64.logand pflags 0xF000_0000L)
              in
              if
                (not (result_ok && mem_ok && flags_ok))
                && not (Hashtbl.mem failures f.Form.name)
              then
                Hashtbl.replace failures f.Form.name
                  (Printf.sprintf
                     "inputs %s nzcv %Lx seed %Lx: native %Lx:%Lx flags %Lx, \
                      model %Lx:%Lx flags %Lx%s"
                     (String.concat "," (List.map (Printf.sprintf "%Lx") ins))
                     (Int64.shift_right_logical nz 28)
                     seed hi lo
                     (Int64.shift_right_logical flags 28)
                     phi plo
                     (Int64.shift_right_logical pflags 28)
                     (if mem_ok then "" else " (memory differs)")))
      | _ -> Hashtbl.replace failures f.Form.name "unreadable harness output")
    vectors results;
  let failed = Hashtbl.length failures in
  List.iter
    (fun (f : Form.t) ->
      match Hashtbl.find_opt failures f.Form.name with
      | Some why -> Printf.printf "FAIL %s: %s\n" f.Form.name why
      | None -> ())
    forms;
  List.iter (Printf.printf "FAIL %s\n") address_failures;
  Printf.printf
    "%d of %d forms agree; address forms (adrp, add :lo12:) at %d targets: %s\n"
    (List.length forms - failed)
    (List.length forms) (List.length addresses)
    (if address_failures = [] && addresses <> [] then "agree" else "DISAGREE");
  let failed =
    failed + List.length address_failures + if addresses = [] then 1 else 0
  in
  match mutation with
  | None -> exit (if failed = 0 then 0 else 1)
  | Some _ ->
      (* a mutation must be caught *)
      if failed > 0 then (
        print_endline "mutation detected";
        exit 0)
      else (
        print_endline "MUTATION SURVIVED";
        exit 1)
