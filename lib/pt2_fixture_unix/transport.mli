(** Fetching one URL into a file. The caller verifies what arrives; a transport
    is trusted with nothing but delivering bytes, so a redirect, a proxy or a
    truncated body changes the digest, never the expected pin. *)

type t = url:string -> dest:string -> (unit, string) result
(** Write the response body to [dest]. A partial or failed transfer is an
    [Error]; the caller discards [dest]. *)

val curl : t
(** [curl] with HTTPS only (also across redirects), at most five redirects, and
    failure on an HTTP error status. *)
