# provenance_utils.jl — what code produced this file?
#
# Caller must provide: using SHA.
#
# Banks live outside git, so a bank JSON's only tie to the code that wrote it is
# what the driver records into the payload.  Two quantities do that job, and
# they are not interchangeable:
#
#   git_sha   — the commit.  Cheap, human-readable, and NOT a description of the
#               code that ran: the revision branches here are permanently dirty,
#               so a mid-run edit leaves `git rev-parse HEAD` untouched.
#   code_sha  — SHA-256 over the contents of a named source list.  Moves exactly
#               where the SHA does not.
#
# pipeline/boundary_scan.jl carries its own copies of `git_provenance` and a
# specialised `scan_code_fingerprint` over SCAN_SOURCE_FILES, and additionally
# matches on the result to decide `--resume` skippability.  That file wrote the
# submitted scans, so it is deliberately left alone here; fold it onto these
# helpers the next time it is opened for another reason.

const PROVENANCE_REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

"""
    git_provenance() -> (sha, dirty)

`git rev-parse HEAD` for the repo this file lives in, plus whether the working
tree carries uncommitted changes.  Both degrade to a safe answer rather than
throwing: no git, no repo, or a detached//broken state gives `("", true)` — an
empty SHA and "assume dirty", which is the conservative reading.
"""
function git_provenance()
    sha = try
        strip(read(`git -C $(PROVENANCE_REPO_ROOT) rev-parse HEAD`, String))
    catch
        ""
    end
    dirty = try
        !isempty(strip(read(`git -C $(PROVENANCE_REPO_ROOT) status --porcelain`, String)))
    catch
        true
    end
    return (sha=sha, dirty=dirty)
end

"""
    code_fingerprint(rel_paths) -> String

SHA-256 over the contents of `rel_paths` (repo-relative), truncated to 16 hex
chars.  The relative path is hashed alongside the bytes, so renaming a file
moves the fingerprint even when its contents do not.

A path that does not exist hashes as empty rather than throwing, so a list may
name a file that is absent on some branches — over-inclusion is the safe
direction here.  Under-inclusion is the failure that matters: it lets two
different codebases produce the same fingerprint.
"""
function code_fingerprint(rel_paths::AbstractVector{<:AbstractString})
    ctx = SHA.SHA256_CTX()
    for rel in rel_paths
        path = joinpath(PROVENANCE_REPO_ROOT, rel)
        SHA.update!(ctx, codeunits(rel))
        SHA.update!(ctx, isfile(path) ? read(path) : UInt8[])
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end
