#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <igl/read_triangle_mesh.h>
#include <igl/adjacency_list.h>
#include <igl/per_vertex_normals.h>
#include <igl/AABB.h>

#include <vector>
#include <queue>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
using namespace std::chrono;
using Scalar = double;
using Vec3   = Eigen::Vector3d;
using Vec3i  = Eigen::Vector3i;
using Mat3   = Eigen::Matrix3d;
using SpMat  = Eigen::SparseMatrix<double>;
using Trip   = Eigen::Triplet<double>;

static const Scalar EPS = 1e-12;
static const int NUM_RAYS = 500; // Fibonacci rays


bool readOBJ_with_color(
    const std::string &filename,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &F,
    std::vector<Vec3> &colors) // r,g,b
{
    std::ifstream in(filename);
    if(!in.is_open()){ std::cerr<<"cannot open "<<filename<<"\n"; return false; }

    std::vector<Vec3> verts;
    std::vector<Vec3i> faces;
    colors.clear();

    std::string line;
    while(std::getline(in,line)){
        if(line.empty()) continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if(token=="v"){
            double x,y,z,r=1.0,g=1.0,b=1.0;
            ss >> x >> y >> z;
            // 頂点カラーがあれば読み込む
            if(!(ss >> r >> g >> b)){
                r=g=b=1.0; // 無ければ白
            }
            verts.push_back(Vec3(x,y,z));
            colors.push_back(Vec3(r,g,b));
        } else if(token=="f"){
            int i0,i1,i2;
            char slash; // テクスチャ/法線は無視
            std::string s0,s1,s2;
            ss >> s0 >> s1 >> s2;
            // v/vt/vn の形式でも v のみでも対応
            auto parse_idx=[&](const std::string &s){
                std::stringstream st(s);
                int idx; st >> idx;
                return idx-1; // OBJは1始まり
            };
            i0 = parse_idx(s0);
            i1 = parse_idx(s1);
            i2 = parse_idx(s2);
            faces.push_back(Vec3i(i0,i1,i2));
        }
    }

    // Eigen行列にコピー
    int n=verts.size();
    V.resize(n,3);
    for(int i=0;i<n;++i) V.row(i) = verts[i];

    int m=faces.size();
    F.resize(m,3);
    for(int i=0;i<m;++i) F.row(i) = faces[i];

    return true;
}

void set_is_joint_from_vertex_color(
    const Eigen::MatrixXd& V,
    const std::vector<Vec3>& colors, // r,g,b
    std::vector<char>& is_joint)
{
    is_joint.resize(V.rows(),0);
    for(int i=0;i<V.rows();++i){
        const Vec3 &c = colors[i];
        if(c[0] > 0.9 && c[1] < 0.1 && c[2] < 0.1) 
            is_joint[i] = 1; // 赤 → joint
        else
            is_joint[i] = 0; // 白またはそれ以外 → body
    }
}

// ============================================================
// OBJ writer with vertex color
// ============================================================
void write_obj(
    const std::string& filename,
    const std::vector<Vec3>& V,
    const std::vector<Vec3i>& F,
    const std::vector<char>& is_joint)
{
    std::ofstream out(filename);
    for(size_t i=0;i<V.size();++i){
        double r = is_joint[i] ? 1.0 : 0.0;
        double g = 0.0;
        double b = is_joint[i] ? 0.0 : 1.0;
        const Vec3 &vi = V[i];
        out << "v "
            << vi(0) << " "
            << vi(1) << " "
            << vi(2) << " "
            << " " << r << " " << g << " " << b << "\n";
    }
    for(const auto &f:F)
        out << "f " << f(0)+1 << " " << f(1)+1 << " " << f(2)+1 << "\n";
}

// ===== Fibonacci hemisphere (z >= 0) =====
Eigen::Vector3d fibonacci_hemisphere(int i, int N)
{
    const double phi = (1.0 + std::sqrt(5.0)) * 0.5; // golden ratio
    double u = (i + 0.5) / N; // z
    double theta = 2.0 * M_PI * i / phi;
    double z = u;
    double r = std::sqrt(std::max(0.0, 1.0 - z*z));

    return Eigen::Vector3d(
        r * std::cos(theta),
        r * std::sin(theta),
        z
    );
}

// ===== Local frame rotation to inward direction =====
Eigen::Vector3d to_inward_direction(
    const Eigen::Vector3d& d,
    const Eigen::Vector3d& n)
{
    Eigen::Vector3d Nz = n.normalized();
    Eigen::Vector3d t =
        (std::abs(Nz.x()) > 0.9 ?
         Eigen::Vector3d(0,1,0) :
         Eigen::Vector3d(1,0,0)).cross(Nz).normalized();
    Eigen::Vector3d b = Nz.cross(t);

    return (d.x()*t + d.y()*b - d.z()*Nz).normalized(); // inward
}

// ===== Shape Radius: opposite surface (2-hit) =====
Scalar compute_shape_radius(
    int vi,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& N,
    const igl::AABB<Eigen::MatrixXd,3>& tree,
    Scalar ray_eps)
{
    Eigen::Vector3d o = V.row(vi) + ray_eps * N.row(vi);

    std::vector<Scalar> diameters;
    diameters.reserve(NUM_RAYS);

    for(int k=0;k<NUM_RAYS;++k)
    {
        Eigen::Vector3d d0 = fibonacci_hemisphere(k, NUM_RAYS);
        Eigen::Vector3d dir = to_inward_direction(d0, N.row(vi));

        igl::Hit<Scalar> h1, h2;
        if(!tree.intersect_ray(V,F,o,dir,h1)) continue;

        Eigen::Vector3d o2 = o + (h1.t + ray_eps)*dir;
        if(tree.intersect_ray(V,F,o2,dir,h2) && h2.t > ray_eps)
            diameters.push_back(h1.t + h2.t);
    }

    if(diameters.empty()) return 0.0;

    std::nth_element(diameters.begin(), diameters.begin()+diameters.size()/2, diameters.end());
    return 0.5 * diameters[diameters.size()/2]; // median
}

// ===== r-ring (geodesic distance) =====
void collect_rring(
    int v,
    const std::vector<std::vector<int>>& adj,
    const Eigen::MatrixXd& V,
    Scalar max_dist,
    std::vector<int>& ring)
{
    std::queue<int> q;
    std::vector<Scalar> dist(V.rows(), std::numeric_limits<Scalar>::max());

    dist[v] = 0.0;
    q.push(v);

    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int nb : adj[u]){
            Scalar d = dist[u] + (V.row(u)-V.row(nb)).norm();
            if(d < max_dist && d < dist[nb]){
                dist[nb] = d;
                q.push(nb);
            }
        }
    }

    ring.clear();
    for(int i=0;i<V.rows();++i)
        if(dist[i] < max_dist)
            ring.push_back(i);
}

// ===== LSRV =====
Scalar compute_lsrv(
    int vi,
    const std::vector<Scalar>& radius,
    const std::vector<std::vector<int>>& adj,
    const Eigen::MatrixXd& V,
    Scalar ring_dist)
{
    std::vector<int> ring;
    collect_rring(vi, adj, V, ring_dist, ring);

    std::vector<Scalar> vals;
    for(int j:ring) if(radius[j] > EPS) vals.push_back(radius[j]);

    if(vals.size()<5) return 0.0;

    Scalar mean = 0.0;
    for(Scalar r:vals) mean += r;
    mean /= (Scalar)vals.size();

    Scalar var = 0.0;
    for(Scalar r:vals){ Scalar d=r-mean; var+=d*d; }
    var /= (Scalar)(vals.size()-1);

    return std::sqrt(var)/mean;
}

double cotangent(const Vec3 &a, const Vec3 &b, const Vec3 &c){
    Vec3 u=b-a; Vec3 v=c-a;
    double area=u.cross(v).norm();
    if(area<EPS) return 0.0;
    double d = u.dot(v)/area;
    if(d < 0) d = 1e-6;
    return d;
}

std::unordered_map<long long,double> buildCotWeights(
    const std::vector<Vec3> &V,
    const std::vector<Vec3i> &F)
{
    auto key = [](int i,int j){ long long a=std::min(i,j),b=std::max(i,j); return (a<<32)|b; };
    std::unordered_map<long long,double> w;
    for(const auto &f:F){
        int i=f[0], j=f[1], k=f[2];
        w[key(i,j)]+=cotangent(V[i],V[j],V[k]);
        w[key(j,k)]+=cotangent(V[j],V[k],V[i]);
        w[key(k,i)]+=cotangent(V[k],V[i],V[j]);
    }
    for(auto &it:w) it.second*=0.5;
    return w;
}
struct Neighbor { int j; double w; };

void build_nbr_1ring(
    const std::vector<Vec3>& V,
    const std::vector<std::vector<int>>& adj,
    const std::unordered_map<long long,double>& wmap,
    std::vector<std::vector<Neighbor>>& nbr)
{
    int n = V.size();
    nbr.assign(n, {});

    auto key = [](int i,int j){
        long long a = std::min(i,j), b = std::max(i,j);
        return (a<<32) | b;
    };

    for(int i=0;i<n;i++){
        for(int j: adj[i]){
            auto it = wmap.find(key(i,j));
            if(it != wmap.end()){
                nbr[i].push_back({j, it->second});
            }
        }
    }
}

void build_nbr1(
    const std::vector<std::vector<int>>& adj,
    const std::unordered_map<long long,double>& w,
    std::vector<std::vector<Neighbor>>& nbr1)
{
    int n = adj.size();
    nbr1.assign(n, {});

    auto key = [](int i,int j){
        long long a = std::min(i,j), b = std::max(i,j);
        return (a<<32)|b;
    };

    for(int i=0;i<n;i++){
        for(int j:adj[i]){
            auto it = w.find(key(i,j));
            if(it == w.end()) continue;
            nbr1[i].push_back({j, it->second});
        }
    }
}
void buildAdjList(
    int n,
    const std::vector<std::vector<Neighbor>>& nbr,
    std::vector<std::vector<int>>& adj)
{
    adj.assign(n, {});
    for(int i = 0; i < n; ++i){
        for(const auto& nb : nbr[i]){
            adj[i].push_back(nb.j);
        }
    }
}


struct GlobalSolver {
    Eigen::SimplicialLLT<SpMat> solver;
    std::vector<int> map;   // full -> reduced
    std::vector<int> rmap;  // reduced -> full
};

void initGlobalSolver(
    int n,
    const std::vector<std::vector<int>>& adj,
    const std::unordered_map<long long,double>& w,
    const std::vector<int>& handle_ids,
    GlobalSolver& GS
){
    std::vector<char> is_handle(n,0);
    for(int h:handle_ids) is_handle[h]=1;

    GS.map.assign(n,-1);
    GS.rmap.clear();

    for(int i=0;i<n;++i){
        if(!is_handle[i]){
            GS.map[i] = GS.rmap.size();
            GS.rmap.push_back(i);
        }
    }

    int m = GS.rmap.size();
    SpMat A(m,m);
    std::vector<Trip> T;

    auto key = [](int i,int j){
        long long a=std::min(i,j), b=std::max(i,j);
        return (a<<32)|b;
    };

    for(int ii=0; ii<m; ++ii){
        int i = GS.rmap[ii];
        double diag = 0.0;

        for(int j:adj[i]){
            auto it = w.find(key(i,j));
            if(it==w.end()) continue;
            double wij = it->second;

            diag += wij;

            if(!is_handle[j]){
                int jj = GS.map[j];
                T.emplace_back(ii, jj, -wij);
            }
        }
        T.emplace_back(ii, ii, diag);
    }

    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    GS.solver.compute(A);
    if(GS.solver.info()!=Eigen::Success){
        std::cerr<<"LLT compute failed\n";
        std::exit(1);
    }
}

void globalStep(
    const GlobalSolver& GS,
    const std::vector<Vec3>& V0,
    const std::vector<std::vector<int>>& adj,
    const std::unordered_map<long long,double>& w,
    const std::vector<Mat3>& R,
    const std::vector<int>& handle_ids,
    const std::vector<Vec3>& handle_pos,
    std::vector<Vec3>& V
){
    int m = GS.rmap.size();
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m,3);

    std::vector<char> is_handle(V.size(),0);
    for(int h:handle_ids) is_handle[h]=1;

    auto key = [](int i,int j){
        long long a=std::min(i,j), b=std::max(i,j);
        return (a<<32)|b;
    };

    for(int ii=0; ii<m; ++ii){
        int i = GS.rmap[ii];

        for(int j:adj[i]){
            auto it = w.find(key(i,j));
            if(it==w.end()) continue;
            double wij = it->second;

            B.row(ii) +=
                (0.5 * wij * (R[i]+R[j]) * (V0[i]-V0[j])).transpose();

                auto it2 = std::find(handle_ids.begin(), handle_ids.end(), j);
                if(it2 != handle_ids.end()){
                    int hid = it2 - handle_ids.begin();
                    B.row(ii) += (wij * handle_pos[hid]).transpose();
                }
                
        }
    }

    Eigen::MatrixXd X = GS.solver.solve(B);

    for(int ii=0; ii<m; ++ii)
        V[GS.rmap[ii]] = X.row(ii);

    for(size_t k=0;k<handle_ids.size();++k)
        V[handle_ids[k]] = handle_pos[k];
}

void build_body_rrings(
    const std::vector<std::vector<int>>& adj,
    const std::vector<char>& is_joint,
    int max_ring, // ← ここで 2-ring, 3-ring などを指定
    std::vector<std::vector<int>>& rrings)
{
    max_ring=5;
    int n = adj.size();
    rrings.assign(n, {});

    for(int i=0; i<n; ++i){
        if(is_joint[i]) continue; // body のみ

        std::queue<std::pair<int,int>> q;
        std::vector<int> dist(n,-1);

        dist[i] = 0;
        q.push({i,0});

        while(!q.empty()){
            auto [v,d] = q.front(); q.pop();
            if(d >= max_ring) continue; // max_ring を超えたら追加しない

            for(int nb : adj[v]){
                if(dist[nb]==-1){
                    dist[nb] = d+1;
                    rrings[i].push_back(nb);
                    q.push({nb,d+1});
                }
            }
        }
    }
}

void localStep_ARAP_RR_integrated(
    const std::vector<Vec3>& V0,
    const std::vector<Vec3>& V,
    const std::vector<std::vector<Neighbor>>& nbr, // 1-ring neighbors
    const std::vector<char>& is_joint,
    const std::vector<std::vector<int>>& rrings,   // body 用 r-ring
    std::vector<Mat3>& R,
    double lambda_body = 0.08,
    double lambda_joint = 0.001)
{
    const int n = V0.size();
    std::vector<Mat3> Rprev = R; // 前ステップの回転固定

    for(int i=0; i<n; ++i){
        // 標準 ARAP 相関行列
        Mat3 Si = Mat3::Zero();
        for(const auto& nb : nbr[i]){
            int j = nb.j;
            double w = nb.w;
            Vec3 p = V0[i] - V0[j];
            Vec3 q = V[i] - V[j];
            Si += w * q * p.transpose();
        }

        // r-ring 平滑化
        Mat3 Srr = Mat3::Zero();
        int cnt = 0;
        const std::vector<int>& neighbors = is_joint[i] ? rrings[i] : rrings[i]; // body: r-ring, joint: 1-ring
        for(int j : neighbors){
            Srr += Rprev[j];
            cnt++;
        }
        if(cnt > 0) Srr /= (double)cnt;

        double lambda = is_joint[i] ? lambda_joint : lambda_body;
        Mat3 Stilde = Si + lambda * Srr;

        // SVD で回転更新
        Eigen::JacobiSVD<Mat3> svd(Stilde, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3 U = svd.matrixU();
        Mat3 Vt = svd.matrixV().transpose();
        if((U*Vt).determinant() < 0) U.col(2) *= -1.0;
        R[i] = U * Vt;
    }
}

double computeARAPEnergy_RR(
    const std::vector<Vec3>& V0,
    const std::vector<Vec3>& V,
    const std::vector<std::vector<Neighbor>>& nbr,
    const std::vector<Mat3>& R,
    const std::vector<char>& is_joint
){
    double E=0.0;
    int n=V.size();
    for(int i=0;i<n;++i)
        for(const auto& nb:nbr[i]){
            Vec3 diff = (V[i]-V[nb.j])-0.5*(R[i]+R[nb.j])*(V0[i]-V0[nb.j]);
            E += nb.w*diff.squaredNorm();
        }
    for(int i=0;i<n;++i){
        if(!is_joint[i]) continue;
        for(const auto& nb:nbr[i]){
            if(nb.j<=i) continue;
            Mat3 dR = R[i]-R[nb.j];
            E += 0.5*dR.squaredNorm();
        }
    }
    return E;
}

void ARAP_RR_2ring(
    const std::vector<Vec3>& V0,
    const std::vector<Vec3i>& F,
    const std::vector<std::vector<int>>& adj,
    std::vector<char>& is_joint,
    const std::vector<int>& handle_ids,
    const std::vector<Vec3>& handle_pos,
    std::vector<Vec3>& V,
    int iters)
{
    auto w = buildCotWeights(V0,F);
    std::vector<std::vector<Neighbor>> nbr;
    build_nbr_1ring(V0,adj,w,nbr);
    std::vector<Mat3> R(V0.size(), Mat3::Identity());

    GlobalSolver GS;
    initGlobalSolver(V0.size(), adj, w, handle_ids, GS);

    // body 2-ring
    double avg_edge=0; int ec=0;
    for(int i=0;i<adj.size();++i)
        for(int j:adj[i]) if(j>i){ avg_edge+=(V0[i]-V0[j]).norm(); ec++; }
    avg_edge/=ec;
    double ring_dist = 8.0*avg_edge;
    std::vector<std::vector<int>> rrings;
    build_body_rrings(adj, is_joint,ring_dist, rrings);
    double totalLocalTime = 0.0;
    double totalGlobalTime = 0.0;
    auto t_arap_begin = high_resolution_clock::now();
    double E_prev=std::numeric_limits<double>::infinity();
    for(int iter=0;iter<iters;++iter){
        auto t_local_begin = high_resolution_clock::now();
        localStep_ARAP_RR_integrated(V0,V,nbr,is_joint,rrings,R);
        auto t_local_end = high_resolution_clock::now();
        totalLocalTime += duration<double>(t_local_end-t_local_begin).count();

        auto t_global_begin = high_resolution_clock::now();
        globalStep(GS,V0,adj,w,R,handle_ids,handle_pos,V);
        auto t_global_end = high_resolution_clock::now();
        totalGlobalTime += duration<double>(t_global_end-t_global_begin).count();

        double E = computeARAPEnergy_RR(V0,V,nbr,R,is_joint);
        double rel=(E_prev-E)/std::max(E_prev,1e-12);
        std::cout<<"iter "<<iter+1<<"  E="<<E<<"  rel="<<rel<<"\n";
        if(rel>=0 && rel<1e-3){ std::cout<<"Converged at iter "<<iter+1<<"\n"; break; }
        E_prev=E;
    }

    auto t_arap_end = high_resolution_clock::now();
    double totalTime = duration<double>(t_arap_end - t_arap_begin).count();

        
    std::cout << "\n--- Timing ---\n";
    std::cout << "ARAP total time    : " << totalTime << " sec\n";
    std::cout << "LocalStep total    : " << totalLocalTime << " sec\n";
    std::cout << "GlobalStep total   : " << totalGlobalTime << " sec\n";
    std::cout << "Other overhead     : " << totalTime - totalLocalTime - totalGlobalTime << " sec\n";
    std::cout << "----------------\n";
}
// ============================================================
// main
// ============================================================
int main(int argc,char** argv)
{
    if(argc<3){ std::cerr<<"usage: in.obj out.obj\n"; return 1; }

    Eigen::MatrixXd Vmat;
    Eigen::MatrixXi Fmat;
    if(!igl::readOBJ(argv[1], Vmat, Fmat)){
        std::cerr<<"cannot read "<<argv[1]<<"\n"; return 1;
    }
    /*
    std::vector<Vec3> colors;
    if(!readOBJ_with_color("horse.obj", Vmat, Fmat, colors)) return 1;
    std::vector<char> is_joint;
    set_is_joint_from_vertex_color(Vmat, colors, is_joint);
    */
    std::vector<Vec3> V(Vmat.rows()), Vdef(Vmat.rows());
    for(int i=0;i<Vmat.rows();++i) V[i]=Vdef[i]=Vmat.row(i);

    std::vector<Vec3i> F(Fmat.rows());
    for(int i=0;i<Fmat.rows();++i) F[i]=Fmat.row(i).cast<int>();

    std::vector<std::vector<int>> adj;
    igl::adjacency_list(Fmat, adj);

    Eigen::MatrixXd N;
    igl::per_vertex_normals(Vmat,Fmat,N);

    igl::AABB<Eigen::MatrixXd,3> tree;
    tree.init(Vmat,Fmat);

    Scalar bbox = (Vmat.colwise().maxCoeff() - Vmat.colwise().minCoeff()).norm();
    Scalar ray_eps = 1e-4*bbox;

    Scalar avg_edge=0; int ec=0;
    for(int i=0;i<adj.size();++i)
        for(int j:adj[i]) if(j>i){ avg_edge += (Vmat.row(i)-Vmat.row(j)).norm(); ec++; }
    avg_edge/=ec;
    Scalar ring_dist = 8.0*avg_edge;
    
    std::vector<char> is_joint(V.size(), 0);
    
    // ===== Fibonacci 2-hit radius =====
    std::vector<Scalar> radius(V.size());
    for(int i=0;i<V.size();++i)
        radius[i] = compute_shape_radius(i,Vmat,Fmat,N,tree,ray_eps);

    // ===== LSRV =====
    std::vector<Scalar> lsrv(V.size());
    for(int i=0;i<V.size();++i) lsrv[i] = compute_lsrv(i,radius,adj,Vmat,ring_dist);



    // --- 上位40% ---
    std::vector<Scalar> sorted = lsrv;
    std::sort(sorted.begin(), sorted.end(), std::greater<Scalar>());

    int K = static_cast<int>(0.4 * sorted.size());
    Scalar T = sorted[K];

    for(int i = 0; i < (int)V.size(); ++i){
        if(lsrv[i] >= T)
            is_joint[i] = 1;
    }
    
    std::vector<int> handle_ids = {105634,29003,61399,64272};
    std::vector<Vec3> handle_pos = {V[105634]+Vec3(0,15,15),V[29003]+Vec3(0,15,15),V[61399],V[64272]};

    Scalar handle_ring_dist = 0.8 * ring_dist; // ← 強さはここで調整

    for(int hid : handle_ids){
        std::vector<int> ring;
        collect_rring(hid, adj, Vmat, handle_ring_dist, ring);
        for(int v : ring)
            is_joint[v] = 0;
    }

        //ARAP
        Vdef = V;
        ARAP_RR_2ring(V,F,adj,is_joint,handle_ids,handle_pos,Vdef,500);
        write_obj(argv[2], Vdef, F, is_joint);

        return 0;
}
