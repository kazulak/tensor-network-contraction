using Sockets

# Include the core contraction methods
include("hybrid_contractor.jl")

function start_daemon(port::Int)
    server = listen(port)
    println("Julia Hybrid Daemon started on port $port. Waiting for jobs...")
    
    # We loop indefinitely to accept client connections
    while true
        conn = accept(server)
        # We spawn an asynchronous task for each connection to support concurrent requests
        @async begin
            try
                job_dir = readline(conn)
                if isempty(job_dir)
                    close(conn)
                    return
                end
                
                # Shutdown signal
                if job_dir == "SHUTDOWN"
                    println("Shutdown request received. Closing daemon.")
                    write(conn, "SHUTTING_DOWN\n")
                    close(conn)
                    close(server)
                    exit(0)
                end
                
                println("Daemon received job request: $job_dir")
                # Run the contraction (this compiles on first call, then executes instantly)
                contract_sliced_hybrid(job_dir)
                
                # Read the result from result.txt
                res_txt = read(joinpath(job_dir, "result.txt"), String)
                write(conn, "SUCCESS\n" * res_txt)
            catch e
                println("Error processing contraction job: $e")
                write(conn, "ERROR: $e\n")
            finally
                close(conn)
            end
        end
    end
end

# CLI entry point
if abspath(PROGRAM_FILE) == @__FILE__
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8080
    start_daemon(port)
end
