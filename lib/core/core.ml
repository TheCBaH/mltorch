(* See core.mli. *)

module Monad = Monad

module Error = struct
  type +'e t = { kind : 'e; backtrace : Printexc.raw_backtrace }

  (* [get_callstack] works with no exception in flight and is independent of
     [record_backtrace], so the detection site is captured for free. 64 frames
     is plenty for these validation/config boundaries. *)
  let make kind = { kind; backtrace = Printexc.get_callstack 64 }

  (* Symbolize the captured stack with [Printexc.Slot.format] (the same text the
     runtime uses for exception backtraces) rather than reading [location]
     fields, so this stays robust across compiler versions. Capped for
     readability.

     The innermost frames are this module's own — [make], then whichever of
     [fail]/[of_option] was used — and the detection site is the first frame
     below them. Do NOT count on a fixed index: inlining decides which of those
     frames survives, and it varies. Observed on 4.14.3, [fail] inlines into
     [of_option] but [of_option] keeps its frame, so a bridged option sits one
     frame deeper than a direct [fail]. Assert relationships between frames,
     never an index — see the two tests in test/native/core_test.ml. *)
  let pp_backtrace ppf bt =
    match Printexc.backtrace_slots bt with
    | None ->
        Format.pp_print_string ppf "(backtrace unavailable; build with -g)"
    | Some slots ->
        (* Keep the innermost frames, dropping any the runtime hides (inlined).
           [Slot.format i] needs the original index, so [mapi] before filtering. *)
        let lines =
          Array.sub slots 0 (min (Array.length slots) 10)
          |> Array.to_list
          |> List.mapi Printexc.Slot.format
          |> List.filter_map Fun.id
        in
        Format.fprintf ppf "@[<v>%a@]"
          (Format.pp_print_list ~pp_sep:Format.pp_print_cut
             Format.pp_print_string)
          lines

  let pp pp_payload ppf e =
    Format.fprintf ppf "@[<v>%a@,@[<v2>detected at:@,%a@]@]" pp_payload e.kind
      pp_backtrace e.backtrace
end

type ('a, 'e) result = ('a, 'e Error.t) Stdlib.result

let fail kind = Error (Error.make kind)
let return x = Ok x
let of_option kind = function Some x -> Ok x | None -> fail kind

let map_error f = function
  | Ok x -> Ok x
  | Error e -> Error { e with Error.kind = f e.Error.kind }

let or_raise pp = function
  | Ok x -> x
  | Error e -> failwith (Fmt.str "%a" pp e.Error.kind)

module Syntax = struct
  let ( let* ) r f = match r with Ok x -> f x | Error e -> Error e
  let ( let+ ) r f = match r with Ok x -> Ok (f x) | Error e -> Error e
  let ( >>= ) = ( let* )
  let ( >>| ) = ( let+ )
end

module Pretty = struct
  let to_string pp v = Fmt.str "%a" pp v
  let pp_none none ppf () = Fmt.string ppf none
  let option_or ~none pp = Fmt.option ~none:(pp_none none) pp
  let result ~ok ~error = Fmt.result ~ok ~error
  let error_kind pp ppf e = pp ppf e.Error.kind
  let core_result ~ok ~error = Fmt.result ~ok ~error:(error_kind error)

  let capture_to_string ?like f =
    let buf = Buffer.create 256 in
    let ppf = Fmt.with_buffer ?like buf in
    f ppf;
    Fmt.flush ppf ();
    Buffer.contents buf
end

module List = struct
  let rec map f = function
    | [] -> Ok []
    | x :: xs -> (
        match f x with
        | Error e -> Error e
        | Ok y -> (
            match map f xs with Error e -> Error e | Ok ys -> Ok (y :: ys)))

  let rec iter f = function
    | [] -> Ok ()
    | x :: xs -> ( match f x with Error e -> Error e | Ok () -> iter f xs)

  let rec fold_left f acc = function
    | [] -> Ok acc
    | x :: xs -> (
        match f acc x with Error e -> Error e | Ok acc' -> fold_left f acc' xs)
end
