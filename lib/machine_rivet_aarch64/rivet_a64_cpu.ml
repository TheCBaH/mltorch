(* Whether this CPU can run what an artifact says it needs. Rivet's AArch64
   target has no separable components, so it will assemble any form; the
   deployment machine is a different question, and it is answered from the
   kernel's own report. A machine whose features cannot be read is refused, not
   assumed. *)

module Feature = Machine_ir.Mir_target.Feature

(* The flag Linux's [/proc/cpuinfo] reports for a feature. *)
let flag = function
  | Feature.Fp -> Some "fp"
  | Feature.Neon -> Some "asimd"
  | Feature.Avx | Feature.Avx2 | Feature.Fma | Feature.Sse2 | Feature.Sse41 ->
      None

let features_line path =
  match open_in path with
  | exception Sys_error _ -> None
  | ic ->
      let rec go () =
        match input_line ic with
        | exception End_of_file -> None
        | line -> (
            match String.index_opt line ':' with
            | Some i when String.trim (String.sub line 0 i) = "Features" ->
                Some
                  (String.split_on_char ' '
                     (String.trim
                        (String.sub line (i + 1) (String.length line - i - 1))))
            | _ -> go ())
      in
      let r = go () in
      close_in ic;
      r

(* The first feature the artifact needs and this CPU lacks, if any. *)
let admit ?(cpuinfo = "/proc/cpuinfo") features =
  match features_line cpuinfo with
  | None -> Error Rivet_a64_refusal.Cpu_unknown
  | Some have ->
      List.fold_left
        (fun acc f ->
          match acc with
          | Error _ -> acc
          | Ok () -> (
              match flag f with
              | None -> Error (Rivet_a64_refusal.Cpu_feature (Feature.name f))
              | Some fl ->
                  if List.mem fl have then Ok ()
                  else Error (Rivet_a64_refusal.Cpu_feature (Feature.name f))))
        (Ok ()) features
