(* The failure to parse a persisted digest ([algorithm:hex]). *)

type t =
  | Missing_separator
  | Invalid_algorithm of Text_error.t
  | Invalid_encoding  (** Not nonempty, even-length, lowercase hexadecimal. *)
  | Unsupported_algorithm
      (** The algorithm is syntactically valid but not registered. *)

let pp ppf = function
  | Missing_separator -> Fmt.string ppf "missing ':' separator"
  | Invalid_algorithm e -> Fmt.pf ppf "invalid algorithm (%a)" Text_error.pp e
  | Invalid_encoding -> Fmt.string ppf "invalid encoding"
  | Unsupported_algorithm -> Fmt.string ppf "unsupported algorithm"
