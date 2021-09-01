/*
 * Demonstrate external client inputing and receiving outputs from a SPDZ process,
 * following the protocol described in https://eprint.iacr.org/2015/1006.pdf.
 *
 * Provides a client to bankers_bonus.mpc program to calculate which banker pays for lunch based on
 * the private value annual bonus. Up to 8 clients can connect to the SPDZ engines running
 * the bankers_bonus.mpc program.
 *
 * Each connecting client:
 * - sends a unique id to identify the client
 * - sends an integer input (bonus value to compare)
 * - sends an integer (0 meaining more players will join this round or 1 meaning stop the round and calc the result).
 *
 * The result is returned authenticated with a share of a random value:
 * - share of winning unique id [y]
 * - share of random value [r]
 * - share of winning unique id * random value [w]
 *   winning unique id is valid if ∑ [y] * ∑ [r] = ∑ [w]
 *
 * No communications security is used.
 *
 * To run with 2 parties / SPDZ engines:
 *   ./Scripts/setup-online.sh to create triple shares for each party (spdz engine).
 *   ./compile.py bankers_bonus
 *   ./Scripts/run-online bankers_bonus to run the engines.
 *
 *   ./bankers-bonus-client.x 123 2 100 0
 *   ./bankers-bonus-client.x 456 2 200 0
 *   ./bankers-bonus-client.x 789 2 50 1
 *
 *   Expect winner to be second client with id 456.
 */

#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "Networking/sockets.h"
#include "Networking/ssl_sockets.h"
#include "Tools/int.h"
#include "Math/Setup.h"
#include "Protocols/fake-stuff.h"
#include "Math/Setup.h"

#include <sodium.h>
#include <iostream>
#include <sstream>
#include <fstream>

//
// Created by wuyuncheng on 20/11/19.
//

#include "math.h"

#define SPDZ_FIXED_PRECISION 16

void send_private_inputs(const std::vector<gfp>& values, vector<ssl_socket*>& sockets, int n_parties)
{
    int num_inputs = values.size();
    octetStream os;
    std::vector< std::vector<gfp> > triples(num_inputs, vector<gfp>(3));
    std::vector<gfp> triple_shares(3);

    // Receive num_inputs triples from SPDZ
    for (int j = 0; j < n_parties; j++)
    {
        os.reset_write_head();
        os.Receive(sockets[j]);

        for (int j = 0; j < num_inputs; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                triple_shares[k].unpack(os);
                triples[j][k] += triple_shares[k];
            }
        }
    }

    // Check triple relations (is a party cheating?)
    for (int i = 0; i < num_inputs; i++)
    {
        if (triples[i][0] * triples[i][1] != triples[i][2])
        {
            cerr << "Incorrect triple at " << i << ", aborting\n";
            exit(1);
        }
    }

    // Send inputs + triple[0], so SPDZ can compute shares of each value
    os.reset_write_head();
    for (int i = 0; i < num_inputs; i++)
    {
        gfp y = values[i] + triples[i][0];
        y.pack(os);
    }
    for (int j = 0; j < n_parties; j++)
        os.Send(sockets[j]);
}

void send_private_batch_shares(std::vector<float> shares, vector<ssl_socket*>& sockets, int n_parties) {

    int number_inputs = shares.size();
    std::vector<long> long_shares(number_inputs);

    // step 1: convert to int or long according to the fixed precision
    for (int i = 0; i < number_inputs; ++i) {
        long_shares[i] = static_cast<int>(round(shares[i] * pow(2, SPDZ_FIXED_PRECISION)));
        cout << "long shares[" << i << "] = " << long_shares[i] << endl;
    }

    // step 2: convert to the gfp value and call send_private_inputs
    // Map inputs into gfp
    vector<gfp> input_values_gfp(number_inputs);
    for (int i = 0; i < number_inputs; i++) {
        // input_values_gfp[i].assign(long_shares[i]);
        bigint::tmp = long_shares[i];
        input_values_gfp[i] = gfpvar(bigint::tmp);
    }

    cout << "Begin send private inputs to the SPDZ engine..." << endl;
    // Run the computation
    send_private_inputs(input_values_gfp, sockets, n_parties);
    cout << "Sent private inputs to each SPDZ engine, waiting for result..." << endl;
}

std::vector<float> receive_result(vector<ssl_socket*>& sockets, int n_parties, int size)
{
    cout << "Receive result from the SPDZ engine" <<endl;
    std::vector<gfp> output_values(size);
    octetStream os;
    for (int i = 0; i < n_parties; i++)
    {
        os.reset_write_head();
        os.Receive(sockets[i]);
        for (int j = 0; j < size; j++)
        {
            gfp value;
            value.unpack(os);
            output_values[j] += value;
        }
    }

    std::vector<float> res_shares(size);

    for (int i = 0; i < size; i++) {
        gfp val = output_values[i];
        bigint aa;
        to_signed_bigint(aa, val);
        long t = aa.get_si();
        //cout<< "i = " << i << ", t = " << t <<endl;
        res_shares[i] = static_cast<float>(t * pow(2, -SPDZ_FIXED_PRECISION));
    }

    return res_shares;
}


int main(int argc, char** argv)
{
    int my_client_id;
    int nparties;
    int finish;
    int port_base = 14000;
    string host_name = "localhost";

    if (argc < 4) {
        cout << "Usage is bankers-bonus-client <client identifier> <number of spdz parties> "
           << "<salary to compare> <finish (0 false, 1 true)> <optional host name, default localhost> "
           << "<optional spdz party port base number, default 14000>" << endl;
        exit(0);
    }

    my_client_id = atoi(argv[1]);
    nparties = atoi(argv[2]);
    finish = atoi(argv[3]);
    if (argc > 4)
        host_name = argv[4];
    if (argc > 5)
        port_base = atoi(argv[5]);

    bigint::init_thread();

    cout<<"Begin setup sockets"<<endl;

    for (int i = 0; i < 10; i++) {
        cout <<" ****** iteration " << i << "******" << endl;

        // Setup connections from this client to each party socket
        vector<int> plain_sockets(nparties);
        vector<ssl_socket*> sockets(nparties);
        ssl_ctx ctx("C" + to_string(my_client_id));
        ssl_service io_service;
        octetStream specification;
        for (int i = 0; i < nparties; i++)
        {
          set_up_client_socket(plain_sockets[i], host_name.c_str(), port_base + i);
          send(plain_sockets[i], (octet*) &my_client_id, sizeof(int));
          sockets[i] = new ssl_socket(io_service, ctx, plain_sockets[i],
                                      "P" + to_string(i), "C" + to_string(my_client_id), true);
          if (i == 0)
          {
            cout << "what is here " << endl;
            specification.Receive(sockets[0]);
          }

          octetStream os;
          os.store(finish);
          os.Send(sockets[i]);
          cout << "set up for " << i << "-th party succeed" << ", sockets = " << sockets[i] << ", port_num = " << port_base + i << endl;
        }
        cout << "Finish setup socket connections to SPDZ engines." << endl;

        //gfp::init_field(specification.get<bigint>());
        int type = specification.get<int>();
        switch (type)
        {
          case 'p':
          {
            gfp::init_field(specification.get<bigint>());
            cerr << "using prime " << gfp::pr() << endl;
            //run<gfp>(salary_value, sockets, nparties);
            break;
          }
          default:
            cerr << "Type " << type << " not implemented";
            exit(1);
        }

        // Map inputs into gfp
        int size = 10;
        vector<float> shares(size);
        for (int i = 0; i < size; i++) {
            shares[i] = my_client_id * 0.1 + i * (-0.1) + 0.0;
            cout << "shares[" << i << "] = " << shares[i] << endl;
        }

        cout << "Finish prepare secret shares " << endl;

        // Run the computation
        send_private_batch_shares(shares,sockets,nparties);

        // Get the result back (client_id of winning client)
        vector<float> result = receive_result(sockets, nparties, size);

        cout << "result = ";
        for (int i = 0; i < size; i++) {
            cout << result[i] << ",";
        }
        cout << endl;

        for (int i = 0; i < nparties; i++)
            delete sockets[i];
    }

    return 0;
}
