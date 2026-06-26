(* See core.mli. *)

module Error = struct
  type +'e t = { kind : 'e; backtrace : Printexc.raw_backtrace }

  (* Symbolize the captured stack with [Printexc.Slot.format] (the same text the
     runtime uses for exception backtraces) rather than reading [location]
     fields, so this stays robust across compiler versions. Frame 0 is
     [Core.fail] itself; the real detection site is frame 1. Capped for
     readability. *)
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

(* [get_callstack] works with no exception in flight and is independent of
   [record_backtrace], so the detection site is captured for free. 64 frames is
   plenty for these validation/config boundaries. *)
let fail kind = Error { Error.kind; backtrace = Printexc.get_callstack 64 }
let return x = Ok x

let map_error f = function
  | Ok x -> Ok x
  | Error e -> Error { e with Error.kind = f e.Error.kind }

module Syntax = struct
  let ( let* ) r f = match r with Ok x -> f x | Error e -> Error e
  let ( let+ ) r f = match r with Ok x -> Ok (f x) | Error e -> Error e
  let ( >>= ) = ( let* )
  let ( >>| ) = ( let+ )
end

module List = struct
  let rec map f = function
    | [] -> Ok []
    | x :: xs -> (
        match f x with
        | Error e -> Error e
        | Ok y -> (
            match map f xs with Error e -> Error e | Ok ys -> Ok (y :: ys)))

  let rec fold_left f acc = function
    | [] -> Ok acc
    | x :: xs -> (
        match f acc x with Error e -> Error e | Ok acc' -> fold_left f acc' xs)
end
