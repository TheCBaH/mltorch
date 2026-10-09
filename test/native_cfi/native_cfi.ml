(* The tools are the cross binutils of the architecture ([prefix] is their
   triple prefix, e.g. ["x86_64-linux-gnu-"], or [""] for the native ones). *)

type arch = {
  prefix : string;
  entry_offset : int;  (** the CFA's offset from the stack pointer on entry *)
  effect : mnemonic:string -> operands:string -> int option;
      (** how an instruction moves the stack pointer down (+) or up (-), if it does *)
  barrier : string -> bool;  (** a control transfer: the epilogue search stops *)
  is_return : string -> bool;
}

let read path = In_channel.with_open_bin path In_channel.input_all

let run command =
  let out = Filename.temp_file "cfi" ".out" in
  let status =
    Sys.command (Printf.sprintf "%s >%s 2>&1" command (Filename.quote out))
  in
  let s = read out in
  Sys.remove out;
  if status = 0 then Ok s else Error (Printf.sprintf "%s: %s" command s)

let ( let* ) = Result.bind

type row = { loc : int; cfa : int }
type fde = { lo : int; hi : int; rows : row list }

let hex s = int_of_string ("0x" ^ s)

(* readelf --debug-dump=frames-interp: one table per FDE. *)
let parse_frames text =
  let fdes = ref [] and cur = ref None in
  let flush () =
    match !cur with
    | Some (lo, hi, rows) -> fdes := { lo; hi; rows = List.rev rows } :: !fdes
    | None -> ()
  in
  List.iter
    (fun line ->
      match Str.bounded_split (Str.regexp "[ \t]+") line 6 with
      | _ :: _ :: _ :: "FDE" :: _ :: pc :: _ when String.length pc > 3 && String.sub pc 0 3 = "pc=" -> (
          flush ();
          match Str.split (Str.regexp_string "..") (String.sub pc 3 (String.length pc - 3)) with
          | [ lo; hi ] -> cur := Some (hex lo, hex hi, [])
          | _ -> cur := None)
      | _ -> (
          match (!cur, Str.split (Str.regexp "[ \t]+") line) with
          | Some (lo, hi, rows), loc :: cfa :: _
            when String.length loc = 16
                 && (match Str.string_match (Str.regexp "^[0-9a-f]+$") loc 0 with b -> b)
                 && String.contains cfa '+' -> (
              match String.split_on_char '+' cfa with
              | [ _; off ] -> (
                  match int_of_string_opt off with
                  | Some off -> cur := Some (lo, hi, { loc = hex loc; cfa = off } :: rows)
                  | None -> ())
              | _ -> ())
          | _ -> ()))
    (String.split_on_char '\n' text);
  flush ();
  List.rev !fdes

type insn = { addr : int; mnemonic : string; operands : string }

let parse_disassembly text =
  List.filter_map
    (fun line ->
      match String.split_on_char '\t' line with
      | a :: _ :: rest_fields when rest_fields <> [] && String.length a > 0 && a.[String.length a - 1] = ':' -> (
          let rest = String.concat " " rest_fields in
          let a = String.trim (String.sub a 0 (String.length a - 1)) in
          match int_of_string_opt ("0x" ^ a) with
          | None -> None
          | Some addr -> (
              let rest = String.trim rest in
              match String.index_opt rest ' ' with
              | Some k ->
                  Some
                    {
                      addr;
                      mnemonic = String.sub rest 0 k;
                      operands = String.trim (String.sub rest k (String.length rest - k));
                    }
              | None -> Some { addr; mnemonic = rest; operands = "" }))
      | _ -> None)
    (String.split_on_char '\n' text)

(* The offset before each instruction of a function, by the instructions
   themselves; at a return the frame the body had comes back. *)
let expected arch insns =
  let arr = Array.of_list insns in
  let n = Array.length arr in
  let before = Array.make n 0 in
  let cur = ref arch.entry_offset in
  for k = 0 to n - 1 do
    before.(k) <- !cur;
    let i = arr.(k) in
    let eff = arch.effect ~mnemonic:i.mnemonic ~operands:i.operands in
    if arch.is_return i.mnemonic then begin
      (* walk back to the start of the epilogue: the earliest releasing
         instruction after the last control transfer or growth *)
      let j = ref (k - 1) and base = ref !cur in
      let stop = ref false in
      while (not !stop) && !j >= 0 do
        let p = arr.(!j) in
        (match arch.effect ~mnemonic:p.mnemonic ~operands:p.operands with
        | Some d when d < 0 -> base := before.(!j)
        | Some _ -> stop := true
        | None -> if arch.barrier p.mnemonic then stop := true);
        decr j
      done;
      cur := !base
    end
    else
      match eff with Some d -> cur := !cur + d | None -> ()
  done;
  before

let check arch ~assembly =
  let dir = Filename.temp_file "cfi" "" in
  Sys.remove dir;
  Unix.mkdir dir 0o700;
  Fun.protect
    ~finally:(fun () -> ignore (Sys.command ("rm -rf " ^ Filename.quote dir)))
    (fun () ->
      let s = Filename.concat dir "m.s" and o = Filename.concat dir "m.o" in
      Out_channel.with_open_bin s (fun oc -> output_string oc assembly);
      let* _ =
        run (Printf.sprintf "%sas -o %s %s" arch.prefix (Filename.quote o) (Filename.quote s))
      in
      let* frames =
        run (Printf.sprintf "%sreadelf --debug-dump=frames-interp %s" arch.prefix (Filename.quote o))
      in
      let* dis =
        run (Printf.sprintf "%sobjdump -d -j .text %s" arch.prefix (Filename.quote o))
      in
      let fdes = parse_frames frames and insns = parse_disassembly dis in
      if fdes = [] then Error "no frame description entries"
      else begin
        let checked = ref 0 in
        let result =
          List.fold_left
            (fun acc f ->
              let* () = acc in
              let mine = List.filter (fun i -> i.addr >= f.lo && i.addr < f.hi) insns in
              let before = expected arch mine in
              let rec go k = function
                | [] -> Ok ()
                | (i : insn) :: rest ->
                    let row =
                      List.fold_left
                        (fun best r -> if r.loc <= i.addr then Some r else best)
                        None f.rows
                    in
                    (match row with
                    | None -> Error (Printf.sprintf "%x: no CFI row" i.addr)
                    | Some r ->
                        incr checked;
                        if r.cfa = before.(k) then go (k + 1) rest
                        else
                          Error
                            (Printf.sprintf "%x %s %s: CFA offset %d, instructions say %d"
                               i.addr i.mnemonic i.operands r.cfa before.(k)))
              in
              go 0 mine)
            (Ok ()) fdes
        in
        Result.map (fun () -> Printf.sprintf "ok (%d functions, %d instructions)" (List.length fdes) !checked) result
      end)
